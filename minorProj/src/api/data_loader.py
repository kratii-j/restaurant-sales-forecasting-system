"""
data_loader.py
~~~~~~~~~~~~~~
Singleton data loader – reads CSVs once on startup, caches DataFrames
in memory, and exposes accessor helpers for every API endpoint.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


class DataStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _read_csv_with_fallback(self, path: Path, sample_name: str):
        if path.exists():
            return pd.read_csv(path)
        sample_path = path.parent / f"sample_{path.name}"
        if sample_path.exists():
            print(f"⚠️  File {path.name} not found. Using sample: {sample_path.name}")
            return pd.read_csv(sample_path)
        raise FileNotFoundError(f"Could not find {path.name} or its sample {sample_path.name}")

    def load(self):
        if self._loaded:
            return
        print("⏳ Loading data ...")

        # Restaurant master
        try:
            self.restaurants = pd.read_csv(RAW_DIR / "restaurant_dataset.csv", encoding="utf-8-sig")
        except FileNotFoundError:
            self.restaurants = pd.read_csv(RAW_DIR / "sample_restaurant_dataset.csv", encoding="utf-8-sig")
            print("⚠️  Using sample restaurant_dataset.csv")

        self.restaurants.columns = (
            self.restaurants.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("\ufeff", "")
        )

        # Daily orders
        self.orders = self._read_csv_with_fallback(RAW_DIR / "daily_orders.csv", "sample_daily_orders.csv")
        self.orders["date"] = pd.to_datetime(self.orders["date"])

        # Remaining
        self.performance = self._read_csv_with_fallback(PROCESSED_DIR / "restaurant_performance.csv", "sample_restaurant_performance.csv")
        self.model_comparison = self._read_csv_with_fallback(PROCESSED_DIR / "model_comparison_summary.csv", "sample_model_comparison_summary.csv")
        self.analysis_summary = self._read_csv_with_fallback(PROCESSED_DIR / "restaurant_analysis_summary.csv", "sample_restaurant_analysis_summary.csv")

        self._precompute()
        self._loaded = True
        print(f"✅ Data loaded: {len(self.restaurants)} restaurants, {len(self.orders):,} daily records")

    # ── precompute ───────────────────────────────────────────
    def _precompute(self):
        # Daily trends (aggregated across all restaurants)
        self.daily_trends = (
            self.orders.groupby("date")
            .agg(
                total_orders=("total_orders", "sum"),
                total_revenue=("total_revenue", "sum"),
                avg_orders=("total_orders", "mean"),
                avg_revenue=("total_revenue", "mean"),
                restaurant_count=("restaurant_id", "nunique"),
            )
            .reset_index()
            .sort_values("date")
        )

        # Risk classification
        if "rmse" in self.performance.columns and "mean_actual" in self.performance.columns:
            self.performance["risk_score"] = (
                self.performance["rmse"] / (self.performance["mean_actual"] + 1)
            )
            self.performance["risk_level"] = self.performance["risk_score"].apply(
                lambda x: "Low" if x < 0.02 else ("Medium" if x < 0.05 else "High")
            )


def get_store() -> DataStore:
    store = DataStore()
    store.load()
    return store
