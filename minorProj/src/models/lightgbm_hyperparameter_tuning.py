import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

class LightGBMHyperparameterTuning:
    def __init__(self, train_path="data/processed/train.csv"):
        self.train_path = Path(train_path)
        self.results_path = Path("data/processed/lightgbm_tuning_results.csv")
        self.best_params_path = Path("data/processed/lightgbm_best_params.csv")

    def load_data(self):
        train = pd.read_csv(self.train_path)
        return train

    def prepare_features(self, df):
        drop_cols = ["restaurant_id", "date", "total_orders", "total_revenue"]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
        X = X.fillna(X.median(numeric_only=True))
        return X

    def run(self):
        train = self.load_data()
        X_train = self.prepare_features(train)
        y_train = train["total_orders"]

        param_grid = {
            'num_leaves': [31, 50],
            'learning_rate': [0.1, 0.05],
            'n_estimators': [100, 200],
            'max_depth': [5, 10, -1]
        }
        model = lgb.LGBMRegressor(random_state=42)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        grid = GridSearchCV(model, param_grid, scoring=scorer, cv=3, verbose=2, n_jobs=-1)
        grid.fit(X_train, y_train)

        results = pd.DataFrame(grid.cv_results_)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.results_path, index=False)
        best_params = pd.DataFrame([grid.best_params_])
        best_params.to_csv(self.best_params_path, index=False)
        print(f"Hyperparameter tuning results saved → {self.results_path}\n")
        print(f"Best parameters saved → {self.best_params_path}\n")
        print("Best parameters:")
        print(grid.best_params_)

if __name__ == "__main__":
    LightGBMHyperparameterTuning().run()
