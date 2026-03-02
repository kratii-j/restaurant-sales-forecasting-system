"""
config_loader.py
~~~~~~~~~~~~~~~~
Centralised helper for loading project configuration.

Usage:
    from src.utils.config_loader import get_config, get_env

    cfg   = get_config()           # returns full config dict from config.yaml
    model = cfg["models"]["xgboost"]

    api_key = get_env("OPENWEATHER_API_KEY")
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

# ── Resolve project root (two levels up from this file) ──────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Load .env once on import ─────────────────────────────────
_env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_env_path)


def get_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    Load and return the YAML configuration as a Python dictionary.

    Parameters
    ----------
    config_path : str | None
        Absolute or relative path to a YAML file.
        Defaults to ``<PROJECT_ROOT>/config/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Retrieve an environment variable loaded from ``.env``.

    Parameters
    ----------
    key : str
        Name of the environment variable.
    default : str | None
        Fallback value if the variable is not set.

    Returns
    -------
    str | None
        The value of the environment variable, or *default*.
    """
    return os.getenv(key, default)


def get_model_params(model_name: str) -> dict[str, Any]:
    """
    Convenience helper that returns hyperparameters for a given model.

    Parameters
    ----------
    model_name : str
        One of ``random_forest``, ``xgboost``, ``lightgbm``.

    Returns
    -------
    dict
        Hyperparameter dictionary ready to unpack into the model constructor.

    Raises
    ------
    KeyError
        If *model_name* is not defined in the configuration.
    """
    config = get_config()
    models = config.get("models", {})

    if model_name not in models:
        available = ", ".join(models.keys())
        raise KeyError(
            f"Model '{model_name}' not found in config. "
            f"Available models: {available}"
        )

    return models[model_name]


# ── Quick sanity check when run directly ─────────────────────
if __name__ == "__main__":
    cfg = get_config()
    print("✅ Config loaded successfully!")
    print(f"   Project : {cfg['project']['name']}")
    print(f"   Models  : {', '.join(cfg['models'].keys())}")
    print(f"   API Port: {get_env('API_PORT', '8000')}")
