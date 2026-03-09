import numpy as np


def calculate_risk_score(point_pred, lower_bound, upper_bound):
    """
    Calculate demand risk based on prediction uncertainty.
    """

    interval_width = upper_bound - lower_bound
    risk_score = interval_width / (point_pred + 1)

    return risk_score


def classify_risk(risk_score):
    """
    Convert numerical risk score into category.
    """

    if risk_score < 0.2:
        return "Low Risk"
    elif risk_score < 0.5:
        return "Medium Risk"
    else:
        return "High Risk"