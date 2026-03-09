import os
import joblib
from datetime import datetime


def save_model(model, filepath):
    """
    Save trained model to disk
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load model from disk
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def save_model_with_timestamp(model, model_name, directory="data/models"):
    """
    Save model with version timestamp
    """
    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl"

    filepath = os.path.join(directory, filename)

    joblib.dump(model, filepath)

    return filepath