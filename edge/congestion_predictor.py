"""
Congestion predictors - queue-backlog forecasting for the offloading engine.

This module provides several interchangeable forecasters, all exposing the
same interface (`predict_congestion(series, silent=True) -> float`), so the
decision engine and experiment harness can swap forecasters without touching
any other code:

* CongestionPredictor       - damped-trend Holt-Winters (the paper's method)
* NaivePersistencePredictor - predicts b(t+1) = b(t)  (no-forecaster ablation)
* SimpleExpSmoothingPredictor - single exponential smoothing, no trend term
* ARIMAPredictor            - low-order ARIMA(p,d,q) via statsmodels

Design notes (Holt-Winters)
----------------------------
* Uses *damped-trend* double exponential smoothing (Holt, 1957 + Gardner 1985)
  with alpha=0.35, beta=0.25, phi=0.88 by default. The damping factor phi
  prevents the trend component from extrapolating unrealistically far, which
  matters for bursty IoT backlog series that are mean-reverting between
  spikes. All three parameters are constructor arguments (not hardcoded
  class constants) so they can be swept in a sensitivity analysis without
  subclassing.

* The final prediction blends the HW one-step forecast (55 %), a short-window
  linear trend (30 %), and the last observed value (15 %). The last-obs anchor
  keeps the forecast grounded when the series is flat.

* Raw HW one-step MAE is comparable to (in some regimes slightly worse than) a
  naive persistence baseline on this workload - expected for mean-reverting
  queues, where a model that tracks the recent trend can overshoot during the
  frequent reversions between bursts. The contribution of HW is not lower
  point-forecast error but *earlier directional signal*: because the trend
  component starts rising before the backlog peaks, the decision engine can
  route tasks to the cloud one slot ahead of the spike, reducing deadline
  violations. See rolling_mae() / the naive & ARIMA predictors below for a
  quantitative, like-for-like comparison used in the ablation study.
"""

from __future__ import annotations
import warnings


# ═══════════════════════════════════════════════════════════════════════════
#  Shared evaluation helper (works for any predictor implementing the
#  predict_congestion(series, silent=True) interface)
# ═══════════════════════════════════════════════════════════════════════════

def rolling_mae_for(predictor, series: list[float], min_history: int) -> dict:
    """One-step-ahead MAE of `predictor` vs naive persistence, computed
    out-of-sample (predictor only ever sees history[:idx])."""
    if len(series) <= min_history:
        return {"model_mae": 0.0, "naive_mae": 0.0, "count": 0}

    model_errors, naive_errors = [], []
    for idx in range(min_history, len(series)):
        history = series[:idx]
        actual  = series[idx]
        pred    = predictor.predict_congestion(history, silent=True)
        model_errors.append(abs(pred - actual))
        naive_errors.append(abs(history[-1] - actual))

    return {
        "model_mae": sum(model_errors) / len(model_errors),
        "naive_mae": sum(naive_errors) / len(naive_errors),
        "count": len(model_errors),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Holt-Winters damped-trend predictor (the paper's proposed method)
# ═══════════════════════════════════════════════════════════════════════════

class CongestionPredictor:

    def __init__(
        self,
        min_history: int = 6,
        alpha: float = 0.35,
        beta: float = 0.25,
        phi: float = 0.88,
        clip: float = 1.35,
        blend_hw: float = 0.55,
        blend_trend: float = 0.30,
        blend_last: float = 0.15,
        verbose_init: bool = True,
    ):
        self.min_history = min_history
        self.ALPHA = alpha
        self.BETA = beta
        self.PHI = phi
        self.CLIP = clip
        self.BLEND_HW = blend_hw
        self.BLEND_TREND = blend_trend
        self.BLEND_LAST = blend_last
        assert abs(blend_hw + blend_trend + blend_last - 1.0) < 1e-6, \
            "Blend weights must sum to 1.0"

        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self._ES = ExponentialSmoothing
            if verbose_init:
                print("[CongestionPredictor] statsmodels Holt-Winters backend loaded.")
        except ImportError:
            self._ES = None
            if verbose_init:
                print("[CongestionPredictor] statsmodels unavailable - using built-in fallback.")

    # ---- core forecast ----

    def _hw_numpy(self, series: list[float]) -> float:
        """Pure-Python damped Holt implementation (no statsmodels required)."""
        L = float(series[0])
        T = float(series[1] - series[0])
        for x in series[1:]:
            Lp, Tp = L, T
            L = self.ALPHA * x + (1 - self.ALPHA) * (Lp + self.PHI * Tp)
            T = self.BETA * (L - Lp) + (1 - self.BETA) * self.PHI * Tp
        return max(0.0, L + self.PHI * T)

    def _hw_statsmodels(self, series: list[float]) -> float | None:
        """Statsmodels ExponentialSmoothing with damped trend - optimized fit."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model  = self._ES(series, trend="add", damped_trend=True, seasonal=None)
                fitted = model.fit(optimized=True, use_brute=False)
            return float(fitted.forecast(1)[0])
        except Exception:
            return None

    def predict_congestion(self, queue_series: list[float], silent: bool = False) -> float:
        """
        Predict the next-slot queue backlog.
        """
        if not queue_series:
            return 0.0

        if len(queue_series) < self.min_history:
            fallback = float(queue_series[-1])
            if not silent:
                print(f"[Predictor] Warm-up ({len(queue_series)} pts) -> last={fallback:.3f}s")
            return fallback

        span = max(queue_series) - min(queue_series)
        if span < 1e-4:
            return float(queue_series[-1])

        # HW forecast
        if self._ES is not None:
            hw_val = self._hw_statsmodels(queue_series)
            if hw_val is None:
                hw_val = self._hw_numpy(queue_series)
        else:
            hw_val = self._hw_numpy(queue_series)
        hw_val = max(0.0, hw_val)

        # short-window linear trend
        last = float(queue_series[-1])
        slope = 0.0
        if len(queue_series) >= 3:
            diffs = [queue_series[-i] - queue_series[-i - 1] for i in range(1, 3)]
            slope = sum(diffs) / len(diffs)
            slope = max(-0.4, min(0.4, slope))
        trend_val = last + slope

        # blend
        pred = self.BLEND_HW * hw_val + self.BLEND_TREND * trend_val + self.BLEND_LAST * last
        pred = max(0.0, pred)
        pred = min(pred, max(queue_series) * self.CLIP)

        if not silent:
            print(
                f"[Predictor] pred={pred:.3f}s  hw={hw_val:.3f}  "
                f"trend={trend_val:.3f}  last={last:.3f}"
            )
        return pred

    # ---- evaluation helper ----

    def rolling_mae(self, series: list[float]) -> dict:
        """One-step-ahead MAE of this predictor vs naive persistence baseline."""
        return rolling_mae_for(self, series, self.min_history)


# ═══════════════════════════════════════════════════════════════════════════
#  Naive persistence predictor - "no forecaster" ablation control
#  (Reviewer-requested ablation: isolates the HW component's contribution by
#   running the *same* decision engine with the simplest possible forecast.)
# ═══════════════════════════════════════════════════════════════════════════

class NaivePersistencePredictor:
    """Predicts next-slot backlog = last observed backlog. No trend, no model."""

    def __init__(self, min_history: int = 1):
        self.min_history = min_history

    def predict_congestion(self, queue_series: list[float], silent: bool = False) -> float:
        if not queue_series:
            return 0.0
        pred = float(queue_series[-1])
        if not silent:
            print(f"[NaivePredictor] pred={pred:.3f}s")
        return pred

    def rolling_mae(self, series: list[float]) -> dict:
        return rolling_mae_for(self, series, self.min_history)


# ═══════════════════════════════════════════════════════════════════════════
#  Single exponential smoothing - simpler forecaster ablation (no trend term)
# ═══════════════════════════════════════════════════════════════════════════

class SimpleExpSmoothingPredictor:
    """Single exponential smoothing: L_t = alpha*b_t + (1-alpha)*L_{t-1}.
    No trend component - included as an intermediate ablation point between
    naive persistence and full damped-trend Holt-Winters."""

    def __init__(self, alpha: float = 0.35, min_history: int = 3):
        self.alpha = alpha
        self.min_history = min_history

    def predict_congestion(self, queue_series: list[float], silent: bool = False) -> float:
        if not queue_series:
            return 0.0
        if len(queue_series) < self.min_history:
            return float(queue_series[-1])
        L = float(queue_series[0])
        for x in queue_series[1:]:
            L = self.alpha * x + (1 - self.alpha) * L
        if not silent:
            print(f"[SESPredictor] pred={L:.3f}s")
        return max(0.0, L)

    def rolling_mae(self, series: list[float]) -> dict:
        return rolling_mae_for(self, series, self.min_history)


# ═══════════════════════════════════════════════════════════════════════════
#  Low-order ARIMA predictor - forecasting-method ablation (R3's ask:
#  compare against a classical alternative forecaster in the same pipeline).
# ═══════════════════════════════════════════════════════════════════════════

class ARIMAPredictor:
    """Low-order ARIMA(p,d,q) one-step-ahead forecaster, refit each call on
    the available history. Order defaults to (1,1,0): a differenced
    autoregressive model appropriate for a non-stationary, bursty backlog
    series without requiring seasonal terms. Falls back to naive persistence
    if the series is too short or the fit fails (e.g. during warm-up or on a
    degenerate/constant series) - this keeps behaviour well-defined for every
    slot of the simulation instead of raising mid-run."""

    def __init__(self, order: tuple[int, int, int] = (1, 1, 0), min_history: int = 8):
        self.order = order
        self.min_history = min_history
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self._ARIMA = ARIMA
        except ImportError:
            self._ARIMA = None
            print("[ARIMAPredictor] statsmodels unavailable - falling back to naive persistence.")

    def predict_congestion(self, queue_series: list[float], silent: bool = False) -> float:
        if not queue_series:
            return 0.0
        if self._ARIMA is None or len(queue_series) < self.min_history:
            return float(queue_series[-1])

        span = max(queue_series) - min(queue_series)
        if span < 1e-4:
            return float(queue_series[-1])

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = self._ARIMA(queue_series, order=self.order)
                fitted = model.fit()
            pred = float(fitted.forecast(1)[0])
            pred = max(0.0, pred)
        except Exception:
            pred = float(queue_series[-1])

        if not silent:
            print(f"[ARIMAPredictor] pred={pred:.3f}s")
        return pred

    def rolling_mae(self, series: list[float]) -> dict:
        return rolling_mae_for(self, series, self.min_history)
