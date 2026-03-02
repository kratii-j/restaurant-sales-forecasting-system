import pandas as pd
from pathlib import Path


class FeatureEngineer:

    def __init__(self, data_path="data/raw/daily_orders.csv"):
        self.data_path = Path(data_path)
        self.output_path = Path("data/processed/featured_orders.csv")

    def load_data(self):
        df = pd.read_csv(self.data_path)

        df.columns = df.columns.str.lower().str.strip()
        df["date"] = pd.to_datetime(df["date"])

        return df

    def create_time_features(self, df):
        df["day_of_week"] = df["date"].dt.dayofweek
        df["week"] = df["date"].dt.isocalendar().week
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        return df

    def create_lag_features(self, df):
        df = df.sort_values(["restaurant_id", "date"])

        for lag in [1, 7, 14]:
            df[f"lag_{lag}"] = (
                df.groupby("restaurant_id")["total_orders"]
                .shift(lag)
            )

        return df

    def create_rolling_features(self, df):
        df["rolling_mean_7"] = (
            df.groupby("restaurant_id")["total_orders"]
            .transform(lambda x: x.rolling(7).mean())
        )

        df["rolling_std_7"] = (
            df.groupby("restaurant_id")["total_orders"]
            .transform(lambda x: x.rolling(7).std())
        )

        return df

    def save(self, df):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Saved featured dataset → {self.output_path}")

    def run(self):
        df = self.load_data()
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)

        self.save(df)


if __name__ == "__main__":
    FeatureEngineer().run()