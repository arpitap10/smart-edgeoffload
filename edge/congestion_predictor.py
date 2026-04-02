"""Congestion predictor used for queue-backlog forecasting."""

import warnings


class CongestionPredictor:
    def __init__(self, min_history: int = 6, clip_multiplier: float = 1.35):
        self.min_history = min_history
        self.clip_multiplier = clip_multiplier
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            self.ExponentialSmoothing = ExponentialSmoothing
            print("[CongestionPredictor] Holt-Winters backend loaded.")
        except ImportError:
            print("[CongestionPredictor] statsmodels unavailable; using moving average fallback.")
            self.ExponentialSmoothing = None

    def predict_congestion(self, queue_series: list[float], silent: bool = False) -> float:
        """
        Predict the next queue backlog value.

        The input is a workload-driven backlog series in seconds, not a random
        placeholder queue. This keeps the forecast tied to the actual simulator.
        """
        if not queue_series:
            return 0.0

        if len(queue_series) < self.min_history:
            fallback = float(queue_series[-1])
            if not silent:
                print(f"[Predictor] Warm-up mode using last backlog={fallback:.3f}s")
            return fallback

        if max(queue_series) - min(queue_series) < 1e-4:
            return float(queue_series[-1])

        if self.ExponentialSmoothing is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = self.ExponentialSmoothing(
                        queue_series,
                        trend="add",
                        damped_trend=True,
                        seasonal=None,
                    )
                    fitted = model.fit(optimized=True)
                hw_pred = float(fitted.forecast(1)[0])
                last = float(queue_series[-1])
                slope = 0.0
                if len(queue_series) >= 4:
                    diffs = [
                        queue_series[-1] - queue_series[-2],
                        queue_series[-2] - queue_series[-3],
                        queue_series[-3] - queue_series[-4],
                    ]
                    slope = sum(diffs) / len(diffs)
                trend_pred = last + slope
                pred = (0.35 * hw_pred) + (0.15 * trend_pred) + (0.50 * last)
                pred = max(0.0, pred)
                pred = min(pred, max(queue_series) * self.clip_multiplier)
                if not silent:
                    print(
                        f"[Predictor] HW pred={pred:.3f}s | "
                        f"actual={queue_series[-1]:.3f}s"
                    )
                return pred
            except Exception as exc:
                if not silent:
                    print(f"[Predictor] HW failed ({exc}); using moving average fallback.")

        window = queue_series[-self.min_history :]
        pred = max(0.0, sum(window) / len(window))
        if not silent:
            print(f"[Predictor] MA pred={pred:.3f}s")
        return pred

    def rolling_mae(self, series: list[float]) -> dict:
        """
        Evaluate one-step-ahead prediction quality against a persistence baseline.
        """
        if len(series) <= self.min_history:
            return {"model_mae": 0.0, "naive_mae": 0.0, "count": 0}

        model_errors = []
        naive_errors = []
        for idx in range(self.min_history, len(series)):
            history = series[:idx]
            actual = series[idx]
            pred = self.predict_congestion(history, silent=True)
            model_errors.append(abs(pred - actual))
            naive_errors.append(abs(history[-1] - actual))

        return {
            "model_mae": sum(model_errors) / len(model_errors),
            "naive_mae": sum(naive_errors) / len(naive_errors),
            "count": len(model_errors),
        }
