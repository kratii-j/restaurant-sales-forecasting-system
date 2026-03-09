import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LinearRegressionBaseline:
    def __init__(self, train_path="data/processed/train.csv", test_path="data/processed/test.csv"):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.results_path = Path("data/processed/linear_regression_results.csv")

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return train, test

    def prepare_features(self, df):
        # Drop columns not useful for regression
        drop_cols = ["restaurant_id", "date", "total_orders", "total_revenue"]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
        # Fill NaNs with median
        X = X.fillna(X.median(numeric_only=True))
        return X

    def run(self):
        train, test = self.load_data()
        X_train = self.prepare_features(train)
        y_train = train["total_orders"]
        X_test = self.prepare_features(test)
        y_test = test["total_orders"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        results = pd.DataFrame({"model": ["Linear Regression"], "rmse": [rmse], "mae": [mae]})
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.results_path, index=False)
        print(f"Linear regression results saved → {self.results_path}\n")
        print(results)

if __name__ == "__main__":
    LinearRegressionBaseline().run()
