"""Congestion Predictor — Holt-Winters forecasting for queue length"""


class CongestionPredictor:

    def __init__(self):
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self.ExponentialSmoothing = ExponentialSmoothing
            print("[CongestionPredictor] Holt-Winters (statsmodels) loaded OK.")
        except ImportError:
            print("[CongestionPredictor] WARNING: statsmodels not available — "
                  "using moving-average fallback.")
            self.ExponentialSmoothing = None

    def predict_congestion(self, queue_series: list, silent: bool = False) -> float:
        """
        Predict next queue length from historical series.

        Args:
            queue_series : list of past queue-length observations
            silent       : if True, suppress per-call print (used in bulk runs;
                           the experiment script handles its own logging)
        Returns:
            predicted next queue length (>= 0)
        """
        if len(queue_series) < 5:
            fallback = queue_series[-1] if queue_series else 0.0
            if not silent:
                print(f"[Predictor] Only {len(queue_series)} pts — "
                      f"using last observed value: {fallback:.2f}")
            return float(fallback)

        if self.ExponentialSmoothing is not None:
            try:
                model   = self.ExponentialSmoothing(
                    queue_series, trend="add", seasonal=None)
                fitted  = model.fit(optimized=True)
                pred    = float(fitted.forecast(1)[0])
                pred    = max(0.0, pred)
                if not silent:
                    print(f"[Predictor] Holt-Winters: {pred:.2f}  "
                          f"(tail={[round(x,1) for x in queue_series[-5:]]})")
                return pred
            except Exception as e:
                if not silent:
                    print(f"[Predictor] HW failed ({e}) — moving average fallback.")

        # moving-average fallback
        window = queue_series[-5:]
        pred   = sum(window) / len(window)
        if not silent:
            print(f"[Predictor] Moving-avg: {pred:.2f}  "
                  f"(window={[round(x,1) for x in window]})")
        return pred