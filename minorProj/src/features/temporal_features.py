"""
temporal_features.py
~~~~~~~~~~~~~~~~~~~~
Step 14 — Temporal Feature Engineering

Builds time-varying features from ``daily_orders.csv`` and
``external_context.csv``.  This module supersedes the basic time /
lag / rolling features in ``feature_engineering.py`` by adding:

    • Enhanced calendar features (quarter, day-of-month, cyclical encoding)
    • Extended lag features (1,2,3,7,14,21,28)
    • Extended rolling-window statistics (mean, std, min, max for 7/14/30)
    • Revenue-derived lag & rolling features
    • Trend / momentum features (order differences, week-over-week change)
    • External-context merge (weather encoding, holiday, event flags)

Outputs
-------
* ``data/processed/temporal_features.csv``
    One row per (restaurant_id, date), ready for model training.

Usage
-----
    python -m src.features.temporal_features      # from project root
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
#  TemporalFeatureEngineer
# ══════════════════════════════════════════════════════════════

class TemporalFeatureEngineer:
    """
    Engineers time-varying features for the restaurant sales
    forecasting pipeline.

    Feature groups
    --------------
    1. **Calendar features**        – day/week/month/quarter + cyclical sin/cos
    2. **Lag features (orders)**    – lag 1,2,3,7,14,21,28
    3. **Lag features (revenue)**   – lag 1,7,14
    4. **Rolling features (orders)**– mean,std,min,max for windows 7,14,30
    5. **Rolling features (revenue)** – mean for windows 7,14
    6. **Trend / momentum**         – day-over-day diff, week-over-week change
    7. **External context**         – weather encoding, holiday/event flags
    """

    # Lag periods for total_orders
    _ORDER_LAGS = [1, 2, 3, 7, 14, 21, 28]

    # Lag periods for total_revenue
    _REVENUE_LAGS = [1, 7, 14]

    # Rolling window sizes
    _ROLLING_WINDOWS = [7, 14, 30]

    # Weather encoding map
    _WEATHER_MAP = {
        "Clear":   0,
        "Cloudy":  1,
        "Rainy":   2,
        "Snowy":   3,
        "Stormy":  4,
        "Unknown": 5,
    }

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.orders_path = self.raw_dir / "daily_orders.csv"
        self.external_path = self.raw_dir / "external_context.csv"
        self.restaurant_path = self.raw_dir / "restaurant_dataset.csv"
        self.output_path = self.processed_dir / "temporal_features.csv"

    # ── data loading ────────────────────────────────────────

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load daily orders, external context, and restaurant city mapping."""
        # Daily orders
        orders = pd.read_csv(self.orders_path)
        orders.columns = orders.columns.str.lower().str.strip()
        orders["date"] = pd.to_datetime(orders["date"])
        orders = orders.sort_values(["restaurant_id", "date"]).reset_index(drop=True)

        # External context
        external = pd.read_csv(self.external_path)
        external.columns = external.columns.str.lower().str.strip()
        external["date"] = pd.to_datetime(external["date"])

        # Restaurant → city mapping (for joining external context)
        resto = pd.read_csv(self.restaurant_path, encoding="utf-8-sig")
        resto.columns = (
            resto.columns.str.lower().str.strip()
            .str.replace(" ", "_").str.replace("\ufeff", "")
        )
        city_map = resto[["restaurant_id", "city"]].drop_duplicates()

        logger.info(
            "Loaded: orders=%d rows, external=%d rows, restaurants=%d",
            len(orders), len(external), len(city_map),
        )
        return orders, external, city_map

    # ── 1. Calendar features ───────────────────────────────

    @staticmethod
    def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rich calendar features from the ``date`` column.

        Creates:
            day_of_week, day_of_month, day_of_year,
            week, month, quarter, year,
            is_weekend, is_month_start, is_month_end,
            week_of_month,
            sin_day_of_week, cos_day_of_week,
            sin_month, cos_month,
            sin_day_of_year, cos_day_of_year
        """
        dt = df["date"].dt

        df["day_of_week"]  = dt.dayofweek            # 0=Mon … 6=Sun
        df["day_of_month"] = dt.day
        df["day_of_year"]  = dt.dayofyear
        df["week"]         = dt.isocalendar().week.astype(int)
        df["month"]        = dt.month
        df["quarter"]      = dt.quarter
        df["year"]         = dt.year

        # Binary flags
        df["is_weekend"]     = df["day_of_week"].isin([5, 6]).astype(int)
        df["is_month_start"] = dt.is_month_start.astype(int)
        df["is_month_end"]   = dt.is_month_end.astype(int)

        # Week of month (1-based)
        df["week_of_month"] = (df["day_of_month"] - 1) // 7 + 1

        # ── Cyclical encoding (sin/cos) ──────────────────────
        # Day of week: period = 7
        df["sin_day_of_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["cos_day_of_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Month: period = 12
        df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
        df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

        # Day of year: period = 365
        df["sin_day_of_year"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["cos_day_of_year"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        return df

    # ── 2. Lag features (orders) ───────────────────────────

    @staticmethod
    def create_order_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for total_orders grouped by restaurant."""
        grp = df.groupby("restaurant_id")["total_orders"]
        for lag in TemporalFeatureEngineer._ORDER_LAGS:
            df[f"orders_lag_{lag}"] = grp.shift(lag)
        return df

    # ── 3. Lag features (revenue) ──────────────────────────

    @staticmethod
    def create_revenue_lag_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for total_revenue grouped by restaurant."""
        grp = df.groupby("restaurant_id")["total_revenue"]
        for lag in TemporalFeatureEngineer._REVENUE_LAGS:
            df[f"revenue_lag_{lag}"] = grp.shift(lag)
        return df

    # ── 4. Rolling features (orders) ───────────────────────

    @staticmethod
    def create_order_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling-window statistics for total_orders:
        mean, std, min, max for each window size.
        """
        for w in TemporalFeatureEngineer._ROLLING_WINDOWS:
            rolled = df.groupby("restaurant_id")["total_orders"]

            df[f"orders_rmean_{w}"] = rolled.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"orders_rstd_{w}"] = rolled.transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
            df[f"orders_rmin_{w}"] = rolled.transform(
                lambda x: x.rolling(w, min_periods=1).min()
            )
            df[f"orders_rmax_{w}"] = rolled.transform(
                lambda x: x.rolling(w, min_periods=1).max()
            )
        return df

    # ── 5. Rolling features (revenue) ──────────────────────

    @staticmethod
    def create_revenue_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean for total_revenue over 7- and 14-day windows."""
        for w in [7, 14]:
            df[f"revenue_rmean_{w}"] = (
                df.groupby("restaurant_id")["total_revenue"]
                .transform(lambda x: x.rolling(w, min_periods=1).mean())
            )
        return df

    # ── 6. Trend / momentum ────────────────────────────────

    @staticmethod
    def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Trend and momentum features:
        - orders_diff_1     : day-over-day change in orders
        - orders_diff_7     : week-over-week change in orders
        - orders_momentum_7 : (current − rolling_mean_7) / rolling_mean_7+1
        - revenue_diff_1    : day-over-day change in revenue
        - orders_ewm_7      : exponential weighted mean (span=7)
        """
        grp_orders  = df.groupby("restaurant_id")["total_orders"]
        grp_revenue = df.groupby("restaurant_id")["total_revenue"]

        # Differences
        df["orders_diff_1"] = grp_orders.diff(1)
        df["orders_diff_7"] = grp_orders.diff(7)
        df["revenue_diff_1"] = grp_revenue.diff(1)

        # Momentum: how far current orders deviate from 7-day mean
        if "orders_rmean_7" in df.columns:
            df["orders_momentum_7"] = (
                (df["total_orders"] - df["orders_rmean_7"])
                / (df["orders_rmean_7"] + 1)
            )

        # Exponential weighted mean
        df["orders_ewm_7"] = grp_orders.transform(
            lambda x: x.ewm(span=7, adjust=False).mean()
        )

        return df

    # ── 7. External context merge ──────────────────────────

    @staticmethod
    def merge_external_context(
        df: pd.DataFrame,
        external: pd.DataFrame,
        city_map: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge external context (weather, holiday, events) onto daily orders
        via restaurant_id → city → external_context.

        Creates:
            is_holiday, event_flag, weather_encoded
        """
        # Add city to orders
        df = df.merge(city_map, on="restaurant_id", how="left")

        # Merge external context on (date, city)
        ext_cols = ["date", "city"]
        merge_cols = ["is_holiday", "event_flag", "weather"]
        available = [c for c in merge_cols if c in external.columns]
        df = df.merge(
            external[ext_cols + available],
            on=["date", "city"],
            how="left",
        )

        # Encode weather as integer
        if "weather" in df.columns:
            df["weather_encoded"] = (
                df["weather"]
                .map(TemporalFeatureEngineer._WEATHER_MAP)
                .fillna(5)
                .astype(int)
            )
            df.drop(columns=["weather"], inplace=True)

        # Convert bool-like strings to int
        for col in ("is_holiday", "event_flag"):
            if col in df.columns:
                df[col] = df[col].map(
                    {True: 1, False: 0, "True": 1, "False": 0}
                ).fillna(0).astype(int)

        # Drop city (used only for joining; city info is in static features)
        if "city" in df.columns:
            df.drop(columns=["city"], inplace=True)

        return df

    # ── full pipeline ───────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Execute all temporal feature engineering steps and save output."""
        orders, external, city_map = self.load()

        print(f"  Input shape: {orders.shape}")

        # 1. Calendar
        orders = self.create_calendar_features(orders)
        print(f"  After calendar features:       {orders.shape[1]} cols")

        # 2. Order lags
        orders = self.create_order_lag_features(orders)
        print(f"  After order lag features:       {orders.shape[1]} cols")

        # 3. Revenue lags
        orders = self.create_revenue_lag_features(orders)
        print(f"  After revenue lag features:     {orders.shape[1]} cols")

        # 4. Order rolling
        orders = self.create_order_rolling_features(orders)
        print(f"  After order rolling features:   {orders.shape[1]} cols")

        # 5. Revenue rolling
        orders = self.create_revenue_rolling_features(orders)
        print(f"  After revenue rolling features: {orders.shape[1]} cols")

        # 6. Trend / momentum
        orders = self.create_trend_features(orders)
        print(f"  After trend features:           {orders.shape[1]} cols")

        # 7. External context
        orders = self.merge_external_context(orders, external, city_map)
        print(f"  After external context merge:   {orders.shape[1]} cols")

        # Save
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        orders.to_csv(self.output_path, index=False)
        logger.info("Saved %d rows × %d cols → %s", *orders.shape, self.output_path)

        # Summary
        self._print_summary(orders)

        return orders

    @staticmethod
    def _print_summary(df: pd.DataFrame) -> None:
        """Print a concise summary of the engineered features."""
        print("\n" + "=" * 65)
        print("TEMPORAL FEATURE ENGINEERING — SUMMARY")
        print("=" * 65)
        print(f"  Rows      : {len(df):,}")
        print(f"  Columns   : {df.shape[1]}")

        # Group features by prefix
        groups = {
            "Original columns":   [],
            "Calendar features":  [],
            "Order lag features":  [],
            "Revenue lag features": [],
            "Order rolling features": [],
            "Revenue rolling features": [],
            "Trend features":     [],
            "External context":   [],
        }

        original_cols = {
            "restaurant_id", "date", "total_orders", "total_revenue",
            "avg_discount", "promotion_flag", "cancellation_rate",
            "avg_delivery_time",
        }

        for col in df.columns:
            if col in original_cols:
                groups["Original columns"].append(col)
            elif col.startswith("sin_") or col.startswith("cos_") or col in (
                "day_of_week", "day_of_month", "day_of_year", "week",
                "month", "quarter", "year", "is_weekend",
                "is_month_start", "is_month_end", "week_of_month",
            ):
                groups["Calendar features"].append(col)
            elif col.startswith("orders_lag_"):
                groups["Order lag features"].append(col)
            elif col.startswith("revenue_lag_"):
                groups["Revenue lag features"].append(col)
            elif col.startswith("orders_r"):
                groups["Order rolling features"].append(col)
            elif col.startswith("revenue_r"):
                groups["Revenue rolling features"].append(col)
            elif col in ("orders_diff_1", "orders_diff_7", "revenue_diff_1",
                         "orders_momentum_7", "orders_ewm_7"):
                groups["Trend features"].append(col)
            elif col in ("is_holiday", "event_flag", "weather_encoded"):
                groups["External context"].append(col)
            else:
                groups["Original columns"].append(col)

        for group_name, cols in groups.items():
            if cols:
                print(f"\n  {group_name} ({len(cols)}):")
                for col in cols:
                    nulls = df[col].isnull().sum()
                    null_pct = 100 * nulls / len(df)
                    dtype = df[col].dtype
                    print(f"    • {col:<28s}  {str(dtype):<10s}  "
                          f"nulls={nulls:>9,} ({null_pct:5.1f}%)")

        print("\n" + "=" * 65)


# ── CLI entry point ──────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    engineer = TemporalFeatureEngineer()
    result = engineer.run()
    print(f"\n✅ Temporal features saved → {engineer.output_path}")
    print(f"   Shape: {result.shape}")
