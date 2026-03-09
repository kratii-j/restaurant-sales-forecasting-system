import pandas as pd
from pathlib import Path

class BehavioralFeatureEngineer:
    def __init__(self, data_path="data/raw/daily_orders.csv"):
        self.data_path = Path(data_path)
        self.output_path = Path("data/processed/restaurant_features.csv")

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df.columns = df.columns.str.lower().str.strip()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def create_behavioral_features(self, df):
        # Promotion usage rate (per restaurant)
        promo_rate = df.groupby("restaurant_id")["promotion_flag"].mean().rename("promotion_usage_rate")
        # Cancellation rate (per restaurant)
        cancel_rate = df.groupby("restaurant_id")["cancellation_rate"].mean().rename("avg_cancellation_rate")
        # Average delivery time (per restaurant)
        avg_delivery = df.groupby("restaurant_id")["avg_delivery_time"].mean().rename("avg_delivery_time")
        # Order frequency (orders per day)
        order_freq = df.groupby("restaurant_id")["total_orders"].mean().rename("avg_orders_per_day")
        # Merge features
        features = pd.concat([promo_rate, cancel_rate, avg_delivery, order_freq], axis=1).reset_index()
        return features

    def save(self, df):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Saved behavioral features → {self.output_path}")

    def run(self):
        df = self.load_data()
        features = self.create_behavioral_features(df)
        self.save(features)

if __name__ == "__main__":
    BehavioralFeatureEngineer().run()
