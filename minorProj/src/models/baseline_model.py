import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

class BaselineModel:
    def __init__(self, train_path="data/processed/train.csv", test_path="data/processed/test.csv"):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.results_path = Path("data/processed/baseline_results.csv")

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return train, test

    def mean_forecast(self, train, test):
        # Predict mean of training target for all test samples
        y_train = train["total_orders"]
        y_test = test["total_orders"]
        y_pred = np.full_like(y_test, y_train.mean(), dtype=np.float64)
        return y_test, y_pred, "Mean Forecast"

    def last_value_forecast(self, train, test):
        # Predict last value from training for all test samples
        y_train = train["total_orders"]
        y_test = test["total_orders"]
        y_pred = np.full_like(y_test, y_train.iloc[-1], dtype=np.float64)
        return y_test, y_pred, "Last Value Forecast"

    def evaluate(self, y_test, y_pred, model_name):
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        return {"model": model_name, "rmse": rmse, "mae": mae}

    def run(self):
        train, test = self.load_data()
        results = []
        # Mean forecast
        y_test, y_pred, name = self.mean_forecast(train, test)
        results.append(self.evaluate(y_test, y_pred, name))
        # Last value forecast
        y_test, y_pred, name = self.last_value_forecast(train, test)
        results.append(self.evaluate(y_test, y_pred, name))
        # Save results
        results_df = pd.DataFrame(results)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.results_path, index=False)
        print(f"Baseline results saved → {self.results_path}\n")
        print(results_df)

if __name__ == "__main__":
    BaselineModel().run()
