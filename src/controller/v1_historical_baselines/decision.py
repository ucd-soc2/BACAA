"""
decision.py -- Carbon-Aware Scale-Down Controller
=============================================================

Responsibilities:
    Receive Transformer forecasts (q50/q90/confidence)
    -> Compute Q_plan
    -> Compute N_req / D_smooth / alpha / D_fuse / N_target
    -> Return the replica target for the current window

External responsibilities:
    - Kubernetes API calls (handled by the outer control loop)
    - Model inference (handled by inference.py)
    - Metrics collection (handled externally)

Usage:
    from decision import DecisionLayer, CILoader

    ci    = CILoader("snapshots_2026-02-10_IE-2025-5_minute.csv")
    layer = DecisionLayer(config)

    # Call once per decision window
    result = layer.step(
        q50          = 95.0,      # Transformer q50 forecast (requests/window)
        q90          = 130.0,     # Transformer q90 forecast
        confidence   = 0.72,      # per-step confidence
        current_pods = 3,         # Current replica count
        current_time = datetime.now(timezone.utc),
    )
    N_actual = result["N_actual"]  # Pass directly to kubectl scale
"""

import math
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. Carbon intensity lookup (load CSV and forward-fill in time)
# ─────────────────────────────────────────────

class CILoader:
    """
    Load a carbon-intensity trace and look up CI by timestamp.

    Usage:
        ci = CILoader("snapshots_2026-02-10_IE-2025-5_minute.csv")
        value = ci.lookup(datetime.now(timezone.utc))
        rho   = ci.normalized(value)   # Normalize to [0, 1]
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
        """Return the latest available CI value at a UTC datetime (forward-fill)."""
        if t.tzinfo is not None:
            t_naive = t.replace(tzinfo=None)
        else:
            t_naive = t

        ts  = np.datetime64(t_naive)
        idx = int(np.searchsorted(self._times, ts, side="right")) - 1
        idx = max(0, min(idx, len(self._values) - 1))
        return float(self._values[idx])

    def normalized(self, ci_value: float) -> float:
        """Normalize CI to [0, 1]: rho_CI = (CI - CI_min) / (CI_max - CI_min)."""
        span = self.ci_max - self.ci_min
        if span < 1e-9:
            return 0.5
        return float(np.clip((ci_value - self.ci_min) / span, 0.0, 1.0))


# ─────────────────────────────────────────────
# 2. Decision layer
# ─────────────────────────────────────────────

class DecisionLayer:
    """
    Carbon-aware replica decision layer.

    Scale-up path:
        N_req = ceil(Q_plan / mu)
        N_actual = N_req          <-- Scale up immediately, without smoothing

    Scale-down path:
        1. With confirmation disabled, scale-down requests enter carbon-aware history fusion directly
        2. History smoothing: D_smooth = sum w_i * N_req_{t-H+i} (completed windows only)
        3. Carbon-aware weight: alpha = clip(alpha_base + k_ci*rho_CI - k_g*r_gap, ...)
        4. Fused scale-down target: D_fuse = alpha*N_req + (1-alpha)*D_smooth
        5. Capacity floor: N_target = max(ceil(D_fuse), N_req)

    Planning equation:
        Q_plan = C*q50 + (1-C)*q90

    Scale-down gap equation:
        r_gap = (N_cur - N_req) / N_cur    (only for scale-down, where N_req < N_cur)

    Example configuration:
        {
            "mu":              13.75,   # calibrated single-pod capacity (requests/window)
            "N_min":           1,
            "N_max":          4,
            "W":               3,      # D_smooth history length
            "hist_beta":       0.9,    # Exponential weighting coefficient (larger values favor recent windows)
            "alpha_base":      0.50,
            "k_ci":            0.20,   # Carbon-intensity coefficient
            "k_g":             0.10,   # Scale-down gap coefficient
            "alpha_min":       0.40,
            "alpha_max":       0.75,
        }
    """

    def __init__(self, config: dict, ci_loader: CILoader):
        # ── Capacity parameters ──
        self.mu    = float(config["mu"])
        self.N_min = int(config["N_min"])
        self.N_max = int(config["N_max"])

        # ── Scale-down parameters ──
        self.alpha_base    = float(config.get("alpha_base", 0.50))
        self.k_ci          = float(config.get("k_ci", 0.20))
        self.k_g           = float(config.get("k_g", 0.10))
        self.alpha_min     = float(config.get("alpha_min", 0.40))
        self.alpha_max     = float(config.get("alpha_max", 0.75))

        # ── History window: exponential weights──
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

        # ── Consecutive scale-in confirmation──
        # scale_in_confirm = 1 disables confirmation; 2 or 3 requires consecutive scale-in decisions.
        self.scale_in_confirm = max(int(config.get("scale_in_confirm_windows", 1)), 1)

        # ── State ──
        self._scale_in_counter = 0

        print(f"[DecisionLayer] Ready.")
        print(f"  mu={self.mu}, N=[{self.N_min},{self.N_max}], "
              f"scale-down confirmation={'disabled' if self.scale_in_confirm <= 1 else f'{self.scale_in_confirm} windows'}")
        print(f"  alpha_base={self.alpha_base}, k_ci={self.k_ci}, "
              f"k_g={self.k_g}, W={self.W}, hist_beta={hist_beta}")

    # ── Internal helpers ──────────────────────────────

    def _n_req(self, q_plan: float) -> int:
        """Map Q_plan to the minimum required replicas: N_req = ceil(Q_plan / mu)."""
        raw = math.ceil(max(q_plan, 0.0) / max(self.mu, 1e-9))
        return int(np.clip(raw, self.N_min, self.N_max))

    def _d_smooth(self, hist_values=None) -> float:
        """Exponentially weighted historical demand: D_smooth = sum w_i * N_req_{t-H+i}."""
        hist = self._hist if hist_values is None else hist_values
        if not hist:
            return float(self.N_min)
        arr = np.array(hist, dtype=float)
        w   = self._hist_weights[-len(arr):]
        w   = w / w.sum()
        return float(np.dot(arr, w))

    def _alpha(self, rho_ci: float, r_gap: float) -> float:
        """Carbon-aware adaptive fusion weight."""
        raw = self.alpha_base + self.k_ci * rho_ci - self.k_g * r_gap
        return float(np.clip(raw, self.alpha_min, self.alpha_max))

    # ── Public interface ────────────────────────────────

    def compute_q_plan(self,
                       q50: float,
                       q90: float,
                       confidence: float) -> float:
        """
        Q_plan = C*q50 + (1-C)*q90

        Equivalent form: Q_plan = q50 + (1-C)*(q90 - q50)
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
        Perform one complete decision per control window.

        Parameters:
            q50          : Transformer q50 forecast (same units as mu)
            q90          : Transformer q90 forecast
            confidence   : Confidence C in [0, 1]
            current_pods : Current replica count N_cur
            current_time : UTC datetime for the CI lookup
            q_plan       : Optional precomputed Q_plan; calculated internally when omitted

        Return a dictionary with the following key fields:
            N_actual        : Final replica target to apply
            N_req           : Minimum replicas implied by the current forecast
            D_smooth        : Smoothed historical demand
            D_fuse          : Fused replica demand
            alpha           : Scale-down fusion weight for this step
            r_gap           : Scale-down gap ratio (N_cur - N_req) / N_cur
            rho_ci          : Normalized CI
            ci_now          : Current raw CI (gCO2/kWh)
            q_plan          : QPS used for capacity planning
            scale_direction : "up" / "down" / "hold"
        """
        # ── 1) Q_plan──
        if q_plan is None:
            q_plan = self.compute_q_plan(q50, q90, confidence)

        # ── 2) N_req──
        N_req = self._n_req(q_plan)

        # ── 3) Compute D_smooth from history before the current decision ──
        # Current N_req is appended after the decision so it is not counted twice
        # through both alpha*N_req and the smoothed historical term.
        past_hist = tuple(self._hist)
        D_smooth = self._d_smooth(past_hist) if past_hist else float(current_pods)

        # ── 4) CI lookup ──
        ci_now = self.ci_loader.lookup(current_time)
        rho_ci = self.ci_loader.normalized(ci_now)

        # ── 5) Decision branches──
        if N_req > current_pods:
            # ── Immediate scale-up without smoothing ──
            self._scale_in_counter = 0
            N_target = N_req
            alpha    = self.alpha_base   # Recorded for diagnostics only
            D_fuse   = N_req
            N_actual = int(np.clip(N_target, self.N_min, self.N_max))
            direction = "up"

        elif N_req == current_pods:
            # ── Hold the current replica count──
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

        # ── 6) Compute r_gap for the output ──
        if N_req < current_pods:
            out_r_gap = float((current_pods - N_req) / max(current_pods, 1))
        else:
            out_r_gap = 0.0

        result = {
            # Primary outputs
            "N_actual":        N_actual,
            # Intermediate values for logging and diagnostics
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
# 3. Example simulation
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys

    ci_path = sys.argv[1] if len(sys.argv) > 1 \
        else "carbon_intensity_ie_2025_5min.csv"

    if not os.path.exists(ci_path):
        print(f"[Error] CI file not found: {ci_path}")
        print("Usage: python3 decision.py <path_to_ci_csv>")
        sys.exit(1)

    # ── Small-demand example configuration (separate from the controller runtime configuration)──────────────────
    config = {
        "mu":             4.0,    # Safe per-Pod capacity (requests/window)
        "N_min":          1,
        "N_max":          4,
        "W":              3,      # History length
        "hist_beta":       0.9,    # Exponential weighting coefficient
        "alpha_base":     0.50,
        "k_ci":           0.20,
        "k_g":            0.10,
        "alpha_min":      0.40,
        "alpha_max":      0.75,
    }

    ci    = CILoader(ci_path)
    layer = DecisionLayer(config, ci)

    # ── Simulate 10 control windows with scale-up, hold, and scale-down scenarios ──
    print("\n" + "=" * 70)
    print("Simulating 10 control windows (scale-down confirmation disabled, mu=4)")
    print("=" * 70)

    # (q50, q90, confidence, current_pods)
    steps = [
        (16.0, 20.0, 0.85, 3),   #  1: high demand path; may scale up immediately
        (16.0, 20.0, 0.85, 3),   #  2: repeated demand; expected hold after reaching target
        (16.0, 20.0, 0.85, 3),   #  3: repeated demand; no scale-down gate is used
        (24.0, 30.0, 0.60, 2),   #  4: N_req=ceil(6.6/4)=2 == cur, hold, Reset the confirmation counter
        (30.0, 38.0, 0.50, 2),   #  5: N_req=ceil(8.5/4)=3 > cur=2, Immediate scale-up to 3
        (32.0, 40.0, 0.55, 3),   #  6: N_req=ceil(9.2/4)=3 == cur, hold
        (12.0, 18.0, 0.50, 3),   #  7: lower demand; immediately enters carbon-aware scale-down fusion
        (10.0, 16.0, 0.45, 3),   #  8: lower demand; immediate scale-down fusion again
        (10.0, 16.0, 0.45, 3),   #  9: lower demand; no consecutive wait
        (32.0, 40.0, 0.55, 1),   # 10: N_req=ceil(9.2/4)=3 > cur=1, Immediate scale-up to 3
    ]

    test_time = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

    print(f"\n{'Step':>3}  {'q50':>6}  {'q90':>6}  {'C':>5}  "
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
    print("Decision-layer simulation completed")
    print("  - Scale-down: enter carbon-aware history fusion without consecutive confirmation")
    print("  - Scale-up: N_req > N_cur -> apply N_req immediately")
    print("  - Hold: N_req == N_cur -> hold")
    print("=" * 70)
