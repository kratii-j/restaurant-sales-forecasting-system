import pandas as pd
from pathlib import Path

class TrainTestSplitter:
    def __init__(self, input_path="data/processed/temporal_features.csv", test_size=0.2):
        self.input_path = Path(input_path)
        self.test_size = test_size
        self.train_path = Path("data/processed/train.csv")
        self.test_path = Path("data/processed/test.csv")

    def load_data(self):
        df = pd.read_csv(self.input_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def split(self, df):
        # Sort by date for time series split
        df = df.sort_values("date")
        split_idx = int(len(df) * (1 - self.test_size))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        return train, test

    def save(self, train, test):
        self.train_path.parent.mkdir(parents=True, exist_ok=True)
        train.to_csv(self.train_path, index=False)
        test.to_csv(self.test_path, index=False)
        print(f"Train set saved → {self.train_path}")
        print(f"Test set saved → {self.test_path}")

    def run(self):
        df = self.load_data()
        train, test = self.split(df)
        self.save(train, test)

if __name__ == "__main__":
    TrainTestSplitter().run()
