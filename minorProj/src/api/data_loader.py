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

    def load(self):
        if self._loaded:
            return
        print("⏳ Loading data into memory …")

        # Restaurant master
        self.restaurants = pd.read_csv(RAW_DIR / "restaurant_dataset.csv", encoding="utf-8-sig")
        self.restaurants.columns = (
            self.restaurants.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace("\ufeff", "")
        )

        # Daily orders
        orders = pd.read_csv(RAW_DIR / "daily_orders.csv")
        orders["date"] = pd.to_datetime(orders["date"])
        self.orders = orders

        # Restaurant performance
        self.performance = pd.read_csv(PROCESSED_DIR / "restaurant_performance.csv")

        # Model comparison
        self.model_comparison = pd.read_csv(PROCESSED_DIR / "model_comparison_summary.csv")

        # Restaurant analysis summary
        self.analysis_summary = pd.read_csv(PROCESSED_DIR / "restaurant_analysis_summary.csv")

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
