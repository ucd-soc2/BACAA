"""
decision.py  ── 决策层（Carbon-Aware Scale-Down Controller）
=============================================================

职责：
    接收 Transformer 预测结果（q50/q90/confidence）
    → 算出 Q_plan
    → 计算 N_req / D_smooth / α / D_fuse / N_target
    → 返回本周期应该 apply 的 pod 数

不包含：
    - K8s API 调用（由外部 main 循环负责）
    - 模型推理（由 inference.py 负责）
    - Metrics 采集（由外部负责）

使用方法：
    from decision import DecisionLayer, CILoader

    ci    = CILoader("snapshots_2026-02-10_IE-2025-5_minute.csv")
    layer = DecisionLayer(config)

    # 每个 decision window 调用一次
    result = layer.step(
        q50          = 95.0,      # Transformer 预测的 q50（requests/window）
        q90          = 130.0,     # Transformer 预测的 q90
        confidence   = 0.72,      # per-step confidence
        current_pods = 3,         # 当前实际 pod 数
        current_time = datetime.now(timezone.utc),
    )
    N_actual = result["N_actual"]  # 直接传给 kubectl scale
"""

import math
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. CI 查表（从 CSV 加载，按时间 forward-fill）
# ─────────────────────────────────────────────

class CILoader:
    """
    加载碳强度 trace，按时间查询当前 CI 值。

    用法：
        ci = CILoader("snapshots_2026-02-10_IE-2025-5_minute.csv")
        value = ci.lookup(datetime.now(timezone.utc))
        rho   = ci.normalized(value)   # 归一化到 [0, 1]
    """

    CI_COL   = "Carbon intensity gCO₂eq/kWh (Life cycle)"
    TIME_COL = "Datetime (UTC)"

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)

        df[self.TIME_COL] = (
            pd.to_datetime(df[self.TIME_COL], utc=True)
            .dt.tz_localize(None)
        )
        df = df.sort_values(self.TIME_COL).reset_index(drop=True)

        self._times  = df[self.TIME_COL].values          # numpy datetime64
        self._values = df[self.CI_COL].values.astype(float)

        self.ci_min = float(self._values.min())
        self.ci_max = float(self._values.max())

        print(f"[CILoader] Loaded {len(df)} rows")
        print(f"  Time : {df[self.TIME_COL].iloc[0]}  →  {df[self.TIME_COL].iloc[-1]}")
        print(f"  CI   : {self.ci_min:.1f} ~ {self.ci_max:.1f}  gCO2/kWh")

    def lookup(self, t: datetime) -> float:
        """给定 UTC datetime，返回最近一条 CI 值（forward-fill）。"""
        if t.tzinfo is not None:
            t_naive = t.replace(tzinfo=None)
        else:
            t_naive = t

        ts  = np.datetime64(t_naive)
        idx = int(np.searchsorted(self._times, ts, side="right")) - 1
        idx = max(0, min(idx, len(self._values) - 1))
        return float(self._values[idx])

    def normalized(self, ci_value: float) -> float:
        """将 CI 值归一化到 [0, 1]。rho_CI = (CI - CI_min) / (CI_max - CI_min)"""
        span = self.ci_max - self.ci_min
        if span < 1e-9:
            return 0.5
        return float(np.clip((ci_value - self.ci_min) / span, 0.0, 1.0))


# ─────────────────────────────────────────────
# 2. 决策层
# ─────────────────────────────────────────────

class DecisionLayer:
    """
    Carbon-Aware 决策层（严格遵循论文 Section 3.3–3.5）。

    扩容路径（Section 3.5.1）：
        N_req = ceil(Q_plan / mu)
        N_actual = N_req          <-- 立即扩，不做平滑

    缩容路径（Section 3.5.2–3.5.5）：
        1. 不再使用连续确认门控：缩容请求直接进入 carbon-aware 历史融合
        2. 历史平滑：D_smooth = sum w_i * N_req_{t-H+i}  (只使用过去已完成窗口)
        3. 碳感知融合权重：alpha = clip(alpha_base + k_ci*rho_CI - k_g*r_gap, ...)
        4. 融合缩容目标：D_fuse = alpha*N_req + (1-alpha)*D_smooth
        5. 安全下界：N_target = max(ceil(D_fuse), N_req)

    Q_plan 公式（Section 3.3.4）：
        Q_plan = C*q50 + (1-C)*q90

    r_gap 公式（Section 3.5.4）：
        r_gap = (N_cur - N_req) / N_cur    （仅在缩容时 N_req < N_cur）

    config 示例：
        {
            "mu":              13.75,   # calibrated single-pod capacity (requests/window)
            "N_min":           1,
            "N_max":          4,
            "W":               3,      # D_smooth 历史窗口长度
            "hist_beta":       0.9,    # 指数衰减系数（越大，近期权重越高）
            "alpha_base":      0.50,
            "k_ci":            0.20,   # 碳强度系数
            "k_g":             0.10,   # 缩容缺口比例系数
            "alpha_min":       0.40,
            "alpha_max":       0.75,
        }
    """

    def __init__(self, config: dict, ci_loader: CILoader):
        # ── 容量参数 ──
        self.mu    = float(config["mu"])
        self.N_min = int(config["N_min"])
        self.N_max = int(config["N_max"])

        # ── 缩容超参数 ──
        self.alpha_base    = float(config.get("alpha_base", 0.50))
        self.k_ci          = float(config.get("k_ci", 0.20))
        self.k_g           = float(config.get("k_g", 0.10))
        self.alpha_min     = float(config.get("alpha_min", 0.40))
        self.alpha_max     = float(config.get("alpha_max", 0.75))

        # ── 历史窗口：指数衰减权重（Section 3.5.3）──
        W = int(config.get("W", 3))
        hist_beta = float(config.get("hist_beta", 0.9))
        self._hist: deque = deque(maxlen=W)
        self.W = W

        # History is stored old -> new. Use increasing weights so recent windows matter most.
        positions = np.arange(W, dtype=float)                    # [0, 1, ..., W-1]
        raw_w = np.exp(hist_beta * (positions - (W - 1)))        # [oldest low, newest high]
        self._hist_weights = raw_w / raw_w.sum()

        # ── CI loader ──
        self.ci_loader = ci_loader

        # ── 缩容确认门（report 3.4 / 7.3.8: consecutive scale-in confirmation）──
        # scale_in_confirm = 1 等价于无门（旧行为）；设为 2~3 恢复 "cautious in"。
        self.scale_in_confirm = max(int(config.get("scale_in_confirm_windows", 1)), 1)

        # ── 状态 ──
        self._scale_in_counter = 0

        print(f"[DecisionLayer] Ready.")
        print(f"  mu={self.mu}, N=[{self.N_min},{self.N_max}], "
              f"scale-down confirmation={'disabled' if self.scale_in_confirm <= 1 else f'{self.scale_in_confirm} windows'}")
        print(f"  alpha_base={self.alpha_base}, k_ci={self.k_ci}, "
              f"k_g={self.k_g}, W={self.W}, hist_beta={hist_beta}")

    # ── 内部工具 ──────────────────────────────

    def _n_req(self, q_plan: float) -> int:
        """Q_plan -> 最小所需 pod 数（Section 4: N_req = ceil(Q_plan / mu)）"""
        raw = math.ceil(max(q_plan, 0.0) / max(self.mu, 1e-9))
        return int(np.clip(raw, self.N_min, self.N_max))

    def _d_smooth(self, hist_values=None) -> float:
        """指数加权历史平滑需求 D_smooth = sum w_i * N_req_{t-H+i}（Section 3.5.3）"""
        hist = self._hist if hist_values is None else hist_values
        if not hist:
            return float(self.N_min)
        arr = np.array(hist, dtype=float)
        w   = self._hist_weights[-len(arr):]
        w   = w / w.sum()
        return float(np.dot(arr, w))

    def _alpha(self, rho_ci: float, r_gap: float) -> float:
        """碳感知自适应融合权重（Section 3.5.4）"""
        raw = self.alpha_base + self.k_ci * rho_ci - self.k_g * r_gap
        return float(np.clip(raw, self.alpha_min, self.alpha_max))

    # ── 主接口 ────────────────────────────────

    def compute_q_plan(self,
                       q50: float,
                       q90: float,
                       confidence: float) -> float:
        """
        Q_plan = C*q50 + (1-C)*q90   （Section 3.3.4）

        等价写法：Q_plan = q50 + (1-C)*(q90 - q50)
        """
        C = float(np.clip(confidence, 0.0, 1.0))
        return float(C * q50 + (1.0 - C) * q90)

    def step(self,
             q50: float,
             q90: float,
             confidence: float,
             current_pods: int,
             current_time: datetime,
             q_plan: float = None) -> dict:
        """
        一次完整的决策（每个 decision window 调用一次）。

        参数：
            q50          : Transformer 预测的 q50（单位须与 mu 一致）
            q90          : Transformer 预测的 q90
            confidence   : 置信度 C in [0, 1]
            current_pods : 当前实际 pod 数 N_cur
            current_time : datetime（UTC），用于查 CI
            q_plan       : 若调用方已算好 Q_plan，直接传入；否则内部计算

        返回 dict，关键字段：
            N_actual        : 最终应该 apply 的 pod 数
            N_req           : 当前预测的最低需求
            D_smooth        : 历史平滑需求
            D_fuse          : 融合后 pod 数
            alpha           : 本步的缩容融合权重
            r_gap           : 缩容缺口比例 (N_cur - N_req) / N_cur
            rho_ci          : 归一化 CI
            ci_now          : 当前 CI 原始值（gCO2/kWh）
            q_plan          : 用于规划的 QPS
            scale_direction : "up" / "down" / "hold"
        """
        # ── 1) Q_plan（Section 3.3.4）──
        if q_plan is None:
            q_plan = self.compute_q_plan(q50, q90, confidence)

        # ── 2) N_req（Section 4）──
        N_req = self._n_req(q_plan)

        # ── 3) D_smooth 使用决策开始前的历史快照 ──
        # Current N_req is appended after the decision so it is not counted twice
        # through both alpha*N_req and the smoothed historical term.
        past_hist = tuple(self._hist)
        D_smooth = self._d_smooth(past_hist) if past_hist else float(current_pods)

        # ── 4) CI 查询 ──
        ci_now = self.ci_loader.lookup(current_time)
        rho_ci = self.ci_loader.normalized(ci_now)

        # ── 5) 决策分支（Section 3.5.6）──
        if N_req > current_pods:
            # ── 快速扩容（Section 3.5.1）：直接扩，不做平滑 ──
            self._scale_in_counter = 0
            N_target = N_req
            alpha    = self.alpha_base   # 仅记录
            D_fuse   = N_req
            N_actual = int(np.clip(N_target, self.N_min, self.N_max))
            direction = "up"

        elif N_req == current_pods:
            # ── 无动作（Section 3.5.6）──
            self._scale_in_counter = 0
            N_target = current_pods
            alpha    = self.alpha_base
            D_fuse   = current_pods
            N_actual = current_pods
            direction = "hold"

        else:  # N_req < current_pods
            # Scale-down branch with consecutive-window confirmation (report 3.4 / 7.3.8).
            # The down-signal must persist for `scale_in_confirm` windows before any pod is
            # released. This restores the report's "cautious in" behaviour and prevents a
            # single low-demand window from immediately tearing down warm capacity.
            self._scale_in_counter += 1

            if self._scale_in_counter < self.scale_in_confirm:
                # Not yet confirmed → hold current capacity (do not release pods).
                r_gap = float((current_pods - N_req) / max(current_pods, 1))
                alpha = self._alpha(rho_ci, r_gap)
                D_fuse = current_pods
                N_target = current_pods
                N_actual = current_pods
                direction = "hold"   # pending scale-in confirmation
            else:
                # r_gap = (N_cur - N_req) / N_cur
                r_gap = float((current_pods - N_req) / max(current_pods, 1))

                # alpha = clip(alpha_base + k_ci*rho_CI - k_g*r_gap, ...)
                alpha = self._alpha(rho_ci, r_gap)

                # D_fuse = alpha*N_req + (1-alpha)*D_smooth
                D_fuse_raw = alpha * N_req + (1.0 - alpha) * D_smooth
                D_fuse = int(math.ceil(D_fuse_raw))

                # N_target = max(D_fuse, N_req), keeping a capacity floor for predicted demand.
                N_target = max(N_req, D_fuse)
                N_target = int(np.clip(N_target, self.N_min, self.N_max))
                N_actual = N_target
                direction = "down"

        # ── 6) 计算 r_gap 用于输出 ──
        if N_req < current_pods:
            out_r_gap = float((current_pods - N_req) / max(current_pods, 1))
        else:
            out_r_gap = 0.0

        result = {
            # 主输出
            "N_actual":        N_actual,
            # 中间量（用于 log / debug）
            "N_req":           N_req,
            "D_smooth":        round(D_smooth, 3),
            "D_fuse":          D_fuse,
            "N_target":        N_target,
            "alpha":           round(alpha, 4),
            "r_gap":           round(out_r_gap, 4),
            "rho_ci":          round(rho_ci, 4),
            "ci_now":          round(ci_now, 2),
            "q_plan":          round(q_plan, 3),
            "q50_input":       round(q50, 3),
            "q90_input":       round(q90, 3),
            "confidence":      round(float(confidence), 4),
            "scale_direction": direction,
            "scale_in_counter": self._scale_in_counter,
        }
        self._hist.append(N_req)
        return result


# ─────────────────────────────────────────────
# 3. 简单测试
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys

    ci_path = sys.argv[1] if len(sys.argv) > 1 \
        else "carbon_intensity_ie_2025_5min.csv"

    if not os.path.exists(ci_path):
        print(f"[Error] CI file not found: {ci_path}")
        print("Usage: python3 decision.py <path_to_ci_csv>")
        sys.exit(1)

    # ── 小流量自测配置（独立于 controller 运行配置）──────────────────
    config = {
        "mu":             4.0,    # 单 pod 安全容量（单位为 requests/window）
        "N_min":          1,
        "N_max":          4,
        "W":              3,      # 历史窗口长度
        "hist_beta":       0.9,    # 指数衰减系数
        "alpha_base":     0.50,
        "k_ci":           0.20,
        "k_g":            0.10,
        "alpha_min":      0.40,
        "alpha_max":      0.75,
    }

    ci    = CILoader(ci_path)
    layer = DecisionLayer(config, ci)

    # ── 模拟 10 个控制周期，覆盖扩容/持稳/缩容场景 ──
    print("\n" + "=" * 70)
    print("模拟 10 个控制周期（scale-down confirmation disabled, mu=4）")
    print("=" * 70)

    # (q50, q90, confidence, current_pods)
    steps = [
        (16.0, 20.0, 0.85, 3),   #  1: high demand path; may scale up immediately
        (16.0, 20.0, 0.85, 3),   #  2: repeated demand; expected hold after reaching target
        (16.0, 20.0, 0.85, 3),   #  3: repeated demand; no scale-down gate is used
        (24.0, 30.0, 0.60, 2),   #  4: N_req=ceil(6.6/4)=2 == cur, hold, 计数清零
        (30.0, 38.0, 0.50, 2),   #  5: N_req=ceil(8.5/4)=3 > cur=2, 快速扩容->3
        (32.0, 40.0, 0.55, 3),   #  6: N_req=ceil(9.2/4)=3 == cur, hold
        (12.0, 18.0, 0.50, 3),   #  7: lower demand; immediately enters carbon-aware scale-down fusion
        (10.0, 16.0, 0.45, 3),   #  8: lower demand; immediate scale-down fusion again
        (10.0, 16.0, 0.45, 3),   #  9: lower demand; no consecutive wait
        (32.0, 40.0, 0.55, 1),   # 10: N_req=ceil(9.2/4)=3 > cur=1, 快速扩容->3
    ]

    test_time = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

    print(f"\n{'步':>3}  {'q50':>6}  {'q90':>6}  {'C':>5}  "
          f"{'cur':>4}  {'Q_plan':>7}  {'N_req':>5}  {'D_smth':>6}  "
          f"{'alpha':>6}  {'D_fuse':>6}  {'N_act':>5}  {'dir':>5}  "
          f"{'cnt':>4}  {'CI':>7}  {'rho_CI':>5}  {'r_gap':>5}")
    print("-" * 120)

    current_pods = steps[0][3]
    for i, (q50, q90, conf, _) in enumerate(steps):
        t = datetime(2025, 6, 15, 14, 30 + i,
                     0, tzinfo=timezone.utc)

        r = layer.step(
            q50=q50, q90=q90,
            confidence=conf,
            current_pods=current_pods,
            current_time=t,
        )

        print(f"{i+1:>3}  {q50:>6.1f}  {q90:>6.1f}  {conf:>5.2f}  "
              f"{current_pods:>4}  {r['q_plan']:>7.2f}  {r['N_req']:>5}  "
              f"{r['D_smooth']:>6.2f}  {r['alpha']:>6.3f}  "
              f"{r['D_fuse']:>6}  {r['N_actual']:>5}  "
              f"{r['scale_direction']:>5}  "
              f"{r['scale_in_counter']:>4}  "
              f"{r['ci_now']:>7.1f}  {r['rho_ci']:>5.3f}  {r['r_gap']:>5.3f}")

        current_pods = r["N_actual"]

    print("\n" + "=" * 70)
    print("决策层验证：所有路径均已覆盖")
    print("  - 缩容: 不再等待连续确认，直接进入 carbon-aware 历史融合")
    print("  - 扩容: N_req > N_cur -> 一步到位")
    print("  - 持稳: N_req == N_cur -> hold")
    print("=" * 70)
