"""
static_features.py
~~~~~~~~~~~~~~~~~~
Step 13 — Static Feature Engineering

Extract and engineer time-invariant features from the restaurant master
dataset (``restaurant_dataset.csv``).  These features capture restaurant
positioning, cuisine diversity, quality signals, and geographic context
that remain constant across the forecasting horizon.

Outputs
-------
* ``data/processed/restaurant_features.csv``
    One row per restaurant with all engineered static features.

Usage
-----
    python -m src.features.static_features          # from project root
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────

def _yes_no_to_bool(series: pd.Series) -> pd.Series:
    """Convert 'Yes'/'No' string columns to 1/0 integers."""
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0}
    return series.map(mapping).fillna(0).astype(int)


def _rating_tier(rating: float) -> str:
    """Bin aggregate rating into descriptive tiers."""
    if rating == 0:
        return "unrated"
    elif rating < 2.5:
        return "poor"
    elif rating < 3.5:
        return "average"
    elif rating < 4.0:
        return "good"
    else:
        return "excellent"


# ══════════════════════════════════════════════════════════════
#  StaticFeatureEngineer
# ══════════════════════════════════════════════════════════════

class StaticFeatureEngineer:
    """
    Engineers time-invariant features from the restaurant master table.

    Feature groups
    --------------
    1. **Boolean flags**      – table booking, online delivery, delivering now
    2. **Cuisine features**   – count, top-cuisine flags, cuisine diversity
    3. **Cost features**      – log-cost, cost per price-range bracket
    4. **Rating features**    – rating tier, rating–vote interaction
    5. **Popularity features**– log-votes, votes-per-cuisine
    6. **Geographic features**– city encoding (frequency), country code grouping
    """

    # Most common individual cuisines to create binary flags for
    _TOP_CUISINES = [
        "North Indian",
        "Chinese",
        "Fast Food",
        "Cafe",
        "Bakery",
        "Mughlai",
        "Street Food",
        "South Indian",
        "Italian",
        "Continental",
    ]

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.restaurant_path = self.raw_dir / "restaurant_dataset.csv"
        self.output_path = self.processed_dir / "restaurant_features.csv"

    # ── data loading ────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Load restaurant master data and normalise column names."""
        df = pd.read_csv(self.restaurant_path, encoding="utf-8-sig")
        df.columns = (
            df.columns
            .str.lower()
            .str.strip()
            .str.replace(" ", "_")
            .str.replace("\ufeff", "")
        )
        logger.info("Loaded %d restaurants from %s", len(df), self.restaurant_path)
        return df

    # ── individual feature groups ───────────────────────────

    @staticmethod
    def engineer_boolean_flags(df: pd.DataFrame) -> pd.DataFrame:
        """Convert Yes/No columns to integer flags."""
        for col in ("has_table_booking", "has_online_delivery", "is_delivering_now"):
            if col in df.columns:
                df[col] = _yes_no_to_bool(df[col])
        # switch_to_order_menu has only one value ("No") — drop it
        if "switch_to_order_menu" in df.columns:
            df.drop(columns=["switch_to_order_menu"], inplace=True)
        return df

    @staticmethod
    def engineer_cuisine_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        From the comma-separated ``cuisines`` column, derive:
        - ``cuisine_count``     : number of cuisines offered
        - ``has_<cuisine>``     : binary flag for each top cuisine
        - ``cuisine_diversity`` : Shannon entropy across cuisine flags
        """
        # cuisine count
        df["cuisine_count"] = (
            df["cuisines"]
            .fillna("")
            .apply(lambda x: len([c.strip() for c in x.split(",") if c.strip()]))
        )

        # binary flags for top cuisines
        cuisines_lower = df["cuisines"].fillna("").str.lower()
        for cuisine in StaticFeatureEngineer._TOP_CUISINES:
            col_name = f"has_{re.sub(r'[^a-z0-9]+', '_', cuisine.lower())}"
            df[col_name] = cuisines_lower.str.contains(
                re.escape(cuisine.lower()), regex=True
            ).astype(int)

        # cuisine diversity (simple entropy proxy: -Σ p·log(p))
        cuisine_flag_cols = [
            c for c in df.columns if c.startswith("has_") and c != "has_table_booking"
            and c != "has_online_delivery"
        ]
        cuisine_flags = df[cuisine_flag_cols].values
        row_sums = cuisine_flags.sum(axis=1, keepdims=True)
        # avoid log(0) — set zero-sum rows to 1
        row_sums = np.where(row_sums == 0, 1, row_sums)
        probs = cuisine_flags / row_sums
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.nansum(
                np.where(probs > 0, probs * np.log2(probs), 0), axis=1
            )
        df["cuisine_diversity"] = np.abs(np.round(entropy, 4))

        return df

    @staticmethod
    def engineer_cost_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive cost-related features:
        - ``log_cost``               : log(1 + average_cost_for_two)
        - ``cost_per_price_range``   : average_cost_for_two / price_range
        - ``is_high_cost``           : 1 if cost > 75th percentile
        """
        cost_col = "average_cost_for_two"
        if cost_col not in df.columns:
            return df

        df["log_cost"] = np.log1p(df[cost_col].fillna(0).clip(lower=0))

        # cost normalised by price range
        df["cost_per_price_range"] = np.where(
            df["price_range"] > 0,
            df[cost_col] / df["price_range"],
            0,
        )

        # binary flag for expensive restaurants
        q75 = df[cost_col].quantile(0.75)
        df["is_high_cost"] = (df[cost_col] > q75).astype(int)

        return df

    @staticmethod
    def engineer_rating_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive rating-related features:
        - ``rating_tier``            : categorical (unrated/poor/average/good/excellent)
        - ``is_unrated``             : 1 if aggregate_rating == 0
        - ``rating_x_votes``         : interaction (rating × log-votes)
        """
        if "aggregate_rating" not in df.columns:
            return df

        df["rating_tier"] = df["aggregate_rating"].apply(_rating_tier)
        df["is_unrated"] = (df["aggregate_rating"] == 0).astype(int)

        if "votes" in df.columns:
            log_votes = np.log1p(df["votes"].fillna(0).clip(lower=0))
            df["rating_x_votes"] = df["aggregate_rating"] * log_votes

        return df

    @staticmethod
    def engineer_popularity_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        - ``log_votes``             : log(1 + votes)
        - ``votes_per_cuisine``     : votes / cuisine_count
        - ``is_popular``            : 1 if votes > 75th percentile
        """
        if "votes" not in df.columns:
            return df

        df["log_votes"] = np.log1p(df["votes"].fillna(0).clip(lower=0))

        if "cuisine_count" in df.columns:
            df["votes_per_cuisine"] = np.where(
                df["cuisine_count"] > 0,
                df["votes"] / df["cuisine_count"],
                0,
            )

        q75 = df["votes"].quantile(0.75)
        df["is_popular"] = (df["votes"] > q75).astype(int)

        return df

    @staticmethod
    def engineer_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        - ``city_restaurant_count`` : number of restaurants in the same city
        - ``city_freq``             : proportion of restaurants in the city
        - ``country_group``         : grouped country code (India vs Other)
        """
        if "city" not in df.columns:
            return df

        city_counts = df["city"].value_counts()
        df["city_restaurant_count"] = df["city"].map(city_counts)
        df["city_freq"] = df["city_restaurant_count"] / len(df)

        # Country grouping — India (country_code 1) is dominant
        if "country_code" in df.columns:
            df["is_india"] = (df["country_code"] == 1).astype(int)

        return df

    # ── encode categoricals ─────────────────────────────────

    @staticmethod
    def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Label-encode the ``rating_tier`` column for direct model use.
        Keep ``city`` as-is (will be target-encoded or dropped downstream).
        """
        tier_order = {"unrated": 0, "poor": 1, "average": 2, "good": 3, "excellent": 4}
        if "rating_tier" in df.columns:
            df["rating_tier_encoded"] = df["rating_tier"].map(tier_order).fillna(0).astype(int)
        return df

    # ── column selection ────────────────────────────────────

    @staticmethod
    def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop raw text / high-cardinality columns not needed downstream.
        Keep restaurant_id as the join key.
        """
        drop_cols = [
            "restaurant_name", "address", "locality", "locality_verbose",
            "rating_color", "rating_text", "currency", "cuisines",
        ]
        existing_drops = [c for c in drop_cols if c in df.columns]
        return df.drop(columns=existing_drops)

    # ── full pipeline ───────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Execute all static feature engineering steps and save output."""
        df = self.load()

        # 1. boolean flags
        df = self.engineer_boolean_flags(df)

        # 2. cuisine features
        df = self.engineer_cuisine_features(df)

        # 3. cost features
        df = self.engineer_cost_features(df)

        # 4. rating features
        df = self.engineer_rating_features(df)

        # 5. popularity features
        df = self.engineer_popularity_features(df)

        # 6. geographic features
        df = self.engineer_geographic_features(df)

        # 7. encode categoricals
        df = self.encode_categoricals(df)

        # 8. select final columns
        df = self.select_output_columns(df)

        # Save
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info("Saved %d rows × %d cols → %s", *df.shape, self.output_path)

        # Summary
        self._print_summary(df)

        return df

    @staticmethod
    def _print_summary(df: pd.DataFrame) -> None:
        """Print a concise summary of the engineered features."""
        print("\n" + "=" * 60)
        print("STATIC FEATURE ENGINEERING — SUMMARY")
        print("=" * 60)
        print(f"  Rows     : {len(df):,}")
        print(f"  Columns  : {df.shape[1]}")
        print(f"\n  Feature list ({df.shape[1]} columns):")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            nunique = df[col].nunique()
            nulls = df[col].isnull().sum()
            print(f"    {i:2d}. {col:<30s}  dtype={str(dtype):<10s}  "
                  f"unique={nunique:<6d}  nulls={nulls}")
        print("=" * 60)


# ── CLI entry point ──────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    engineer = StaticFeatureEngineer()
    result = engineer.run()
    print(f"\n✅ Static features saved → {engineer.output_path}")
    print(f"   Shape: {result.shape}")
