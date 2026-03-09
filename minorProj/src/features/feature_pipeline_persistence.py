import joblib
import os


def save_pipeline_object(obj, filepath):
    """
    Save preprocessing objects like encoders, scalers, pipelines
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    joblib.dump(obj, filepath)

    print(f"Pipeline object saved: {filepath}")


def load_pipeline_object(filepath):
    """
    Load preprocessing object
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found")

    obj = joblib.load(filepath)

    print(f"Pipeline object loaded: {filepath}")

    return obj