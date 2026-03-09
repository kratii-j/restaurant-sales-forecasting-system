import json
import os
from datetime import datetime


class ModelRegistry:

    def __init__(self, registry_path="data/models/model_registry.json"):
        self.registry_path = registry_path

        os.makedirs(os.path.dirname(registry_path), exist_ok=True)

        if not os.path.exists(registry_path):
            with open(registry_path, "w") as f:
                json.dump([], f)

    def register_model(
        self,
        model_name,
        model_version,
        metrics,
        hyperparameters,
        features,
        model_path,
    ):
        """
        Register a trained model with metadata
        """

        record = {
            "model_name": model_name,
            "model_version": model_version,
            "training_date": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "features": features,
            "model_path": model_path,
        }

        registry = self._load_registry()

        registry.append(record)

        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=4)

        print("Model registered successfully")

    def _load_registry(self):

        with open(self.registry_path, "r") as f:
            return json.load(f)

    def list_models(self):

        registry = self._load_registry()

        for model in registry:
            print(
                f"{model['model_name']} | version {model['model_version']} | {model['training_date']}"
            )

        return registry