"""Congestion Predictor - Holt-Winters forecasting for queue length"""


class CongestionPredictor:

    def __init__(self):
        self.model = None
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self.ExponentialSmoothing = ExponentialSmoothing
            print("[CongestionPredictor] Holt-Winters (statsmodels) loaded OK.")
        except ImportError:
            print("[CongestionPredictor] WARNING: statsmodels not available — "
                  "will use moving average fallback throughout.")
            self.ExponentialSmoothing = None

    def predict_congestion(self, queue_series: list) -> float:
        """
        Predict next queue length from historical series.

        Args:
            queue_series : list of past queue length observations
        Returns:
            predicted next queue length (>= 0)
        """

        # Need at least 5 points for a meaningful HW fit
        if len(queue_series) < 5:
            fallback = queue_series[-1] if queue_series else 0
            print(f"[Predictor] Only {len(queue_series)} pts — "
                  f"using last observed value: {fallback:.2f}")
            return fallback

        # --- Holt-Winters path ---
        if self.ExponentialSmoothing is not None:
            try:
                model       = self.ExponentialSmoothing(
                    queue_series,
                    trend='add',
                    seasonal=None
                )
                fitted      = model.fit(optimized=True)
                forecast    = fitted.forecast(1)
                predicted   = float(forecast[0])
                predicted   = max(0.0, predicted)

                # FIX: log so the experiment output shows HW is actually running
                print(f"[Predictor] Holt-Winters forecast: {predicted:.2f}  "
                      f"(series tail: {[round(x,1) for x in queue_series[-5:]]})")
                return predicted

            except Exception as e:
                print(f"[Predictor] Holt-Winters failed ({e}) — falling back to moving average.")

        # --- Moving average fallback ---
        window    = queue_series[-5:]
        predicted = sum(window) / len(window)
        print(f"[Predictor] Moving-avg fallback: {predicted:.2f}  "
              f"(window={[round(x,1) for x in window]})")
        return predicted