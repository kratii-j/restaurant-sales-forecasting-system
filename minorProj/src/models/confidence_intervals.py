import numpy as np


def get_prediction_with_intervals(model_point, model_lower, model_upper, features):
    """
    Generate prediction with confidence intervals.

    Parameters:
    model_point : main trained model
    model_lower : lower quantile model
    model_upper : upper quantile model
    features : input feature dataframe/array
    """

    point_pred = model_point.predict(features)
    lower_bound = model_lower.predict(features)
    upper_bound = model_upper.predict(features)

    return point_pred, lower_bound, upper_bound


def validate_interval_coverage(y_true, lower_pred, upper_pred):
    """
    Validate how often the real value falls inside the interval.
    """

    within_interval = (y_true >= lower_pred) & (y_true <= upper_pred)

    coverage = np.mean(within_interval)

    return coverage