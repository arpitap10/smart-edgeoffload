class CongestionPredictor:

    def __init__(self):
        self.model = None
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self.ExponentialSmoothing = ExponentialSmoothing
        except ImportError:
            print("Warning: statsmodels not available, using simple average")
            self.ExponentialSmoothing = None

    def predict_congestion(self, queue_series):
        """
        Predict future congestion using Holt-Winters forecasting

        queue_series : list of past queue lengths
        returns predicted next queue length
        """

        # Need at least a few data points for forecasting
        if len(queue_series) < 5:
            return queue_series[-1] if queue_series else 0

        # Try Holt-Winters first
        if self.ExponentialSmoothing is not None:
            try:
                # Holt-Winters model
                model = self.ExponentialSmoothing(
                    queue_series,
                    trend='add',
                    seasonal=None
                )

                fitted_model = model.fit()

                # Predict next timestep
                forecast = fitted_model.forecast(1)

                predicted_queue = float(forecast[0])

                return max(0, predicted_queue)  # Ensure non-negative

            except Exception as e:
                print(f"Holt-Winters failed: {e}, falling back to moving average")

        # Fallback to simple moving average
        return sum(queue_series[-5:]) / len(queue_series[-5:])