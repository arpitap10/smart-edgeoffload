from statsmodels.tsa.holtwinters import ExponentialSmoothing


class CongestionPredictor:

    def __init__(self):
        pass

    def predict_congestion(self, queue_series):
        """
        Predict future congestion using Holt-Winters forecasting

        queue_series : list of past queue lengths
        returns predicted next queue length
        """

        # Need at least a few data points for forecasting
        if len(queue_series) < 5:
            return queue_series[-1] if queue_series else 0

        try:
            # Holt-Winters model
            model = ExponentialSmoothing(
                queue_series,
                trend='add',
                seasonal=None
            )

            fitted_model = model.fit()

            # Predict next timestep
            forecast = fitted_model.forecast(1)

            predicted_queue = float(forecast[0])

            return predicted_queue

        except Exception:
            # fallback if model fails
            return queue_series[-1]