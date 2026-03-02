import pandas as pd
import numpy as np
from pathlib import Path


class DataValidator:

    def __init__(self, data_path="data/raw"):
        self.data_path = Path(data_path)

        self.restaurant_path = self.data_path / "restaurant_dataset.csv"
        self.orders_path = self.data_path / "daily_orders.csv"
        self.context_path = self.data_path / "external_context.csv"

        self.report = []

    def _log(self, msg):
        print(msg)
        self.report.append(msg)

    def _find_col(self, df, possible_names):
        for c in possible_names:
            if c in df.columns:
                return c
        return None

    def check_missing(self, df, name):
        total = df.isnull().sum().sum()
        self._log(f"[{name}] Missing values: {total}")

    def check_duplicates(self, df):
        id_col = self._find_col(
            df,
            ["restaurant_id", "Restaurant ID", "id", "res_id", "store_id"]
        )

        if not id_col:
            self._log("[Restaurant Dataset] ID column not found → skipped")
            return

        dups = df[id_col].duplicated().sum()
        self._log(f"[Restaurant Dataset] Duplicate IDs ({id_col}): {dups}")

    def check_date_continuity(self, df):
        date_col = self._find_col(df, ["date", "Date", "order_date"])
        id_col = self._find_col(df, ["restaurant_id", "Restaurant ID", "id"])

        if not date_col or not id_col:
            self._log("[Daily Orders] Date continuity skipped (columns missing)")
            return

        df[date_col] = pd.to_datetime(df[date_col])

        broken = 0
        for _, g in df.groupby(id_col):
            full_range = pd.date_range(g[date_col].min(), g[date_col].max())
            if len(full_range) != len(g):
                broken += 1

        self._log(f"[Daily Orders] Restaurants with missing dates: {broken}")

    def check_outliers(self, df, possible_cols, name):
        col = self._find_col(df, possible_cols)

        if not col:
            self._log(f"[{name}] Column not found → skipped outlier check")
            return

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        self._log(f"[{name}] Outliers in {col}: {outliers}")

    def check_ranges(self, df):
        orders_col = self._find_col(df, ["orders", "order_count", "num_orders"])
        revenue_col = self._find_col(df, ["revenue", "sales", "income"])

        if orders_col:
            invalid = (df[orders_col] < 0).sum()
            self._log(f"[Daily Orders] Negative {orders_col}: {invalid}")

        if revenue_col:
            invalid = (df[revenue_col] < 0).sum()
            self._log(f"[Daily Orders] Negative {revenue_col}: {invalid}")

    def run(self):
        self._log("===== DATA QUALITY REPORT =====")

        restaurants = pd.read_csv(self.restaurant_path)
        orders = pd.read_csv(self.orders_path)
        context = pd.read_csv(self.context_path)

        self.check_missing(restaurants, "Restaurant Dataset")
        self.check_missing(orders, "Daily Orders")
        self.check_missing(context, "External Context")

        self.check_duplicates(restaurants)
        self.check_date_continuity(orders)

        self.check_outliers(orders, ["orders", "order_count"], "Daily Orders")
        self.check_outliers(orders, ["revenue", "sales"], "Daily Orders")

        self.check_ranges(orders)

        self.save_report()

    def save_report(self):
        report_path = Path("data/processed/data_quality_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report))

        print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    validator = DataValidator()
    validator.run()