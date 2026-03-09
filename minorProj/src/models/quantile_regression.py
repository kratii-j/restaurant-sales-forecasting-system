import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error


def pinball_loss(y_true, y_pred, alpha):
    # y_true, y_pred: arrays
    diff = y_true - y_pred
    return np.mean(np.maximum(alpha * diff, (alpha - 1) * diff))


class QuantileRegression:
    def __init__(self, train_path='data/processed/train.csv', test_path='data/processed/test.csv'):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.results_path = Path('data/processed/quantile_results.csv')
        self.predictions_dir = Path('data/processed/quantile_predictions')
        self.best_params_path = Path('data/processed/lightgbm_best_params.csv')

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return train, test

    def prepare_features(self, df):
        drop_cols = ["restaurant_id", "date", "total_orders", "total_revenue"]
        X = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
        X = X.fillna(X.median(numeric_only=True))
        return X

    def _load_best_params(self):
        if self.best_params_path.exists():
            try:
                bp = pd.read_csv(self.best_params_path).iloc[0].to_dict()
                # convert numpy types to python
                return {k: int(v) if str(v).isdigit() else v for k, v in bp.items()}
            except Exception:
                return {}
        return {}

    def run(self, alphas=(0.1, 0.5, 0.9)):
        train, test = self.load_data()
        X_train = self.prepare_features(train)
        y_train = train['total_orders'].values
        X_test = self.prepare_features(test)
        y_test = test['total_orders'].values

        best_params = self._load_best_params()
        # defaults
        base_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'num_leaves': 50,
            'max_depth': -1,
            'random_state': 42,
        }
        # update with best_params loaded (if any)
        for k, v in best_params.items():
            # cast to numeric when possible and validate
            try:
                base_params[k] = int(v)
                continue
            except Exception:
                pass
            try:
                base_params[k] = float(v)
            except Exception:
                base_params[k] = v

        # ensure numeric params have valid types/values
        try:
            base_params['learning_rate'] = float(base_params.get('learning_rate', 0.1))
            if base_params['learning_rate'] <= 0:
                base_params['learning_rate'] = 0.1
        except Exception:
            base_params['learning_rate'] = 0.1

        try:
            base_params['n_estimators'] = int(base_params.get('n_estimators', 200))
        except Exception:
            base_params['n_estimators'] = 200

        try:
            base_params['num_leaves'] = int(base_params.get('num_leaves', 50))
        except Exception:
            base_params['num_leaves'] = 50

        try:
            base_params['max_depth'] = int(base_params.get('max_depth', -1))
        except Exception:
            base_params['max_depth'] = -1

        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for alpha in alphas:
            params = base_params.copy()
            params.update({'objective': 'quantile', 'alpha': float(alpha)})

            model = lgb.LGBMRegressor(**params)
            print(f"Training quantile model alpha={alpha} ...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            pinball = pinball_loss(y_test, y_pred, alpha)
            mae = mean_absolute_error(y_test, y_pred)

            results.append({'alpha': alpha, 'pinball_loss': float(pinball), 'mae': float(mae)})

            pred_df = test.copy()
            pred_df['predicted_total_orders'] = y_pred
            out_path = self.predictions_dir / f'quantile_pred_alpha_{alpha:.2f}.csv'
            pred_df.to_csv(out_path, index=False)
            print(f"Saved predictions → {out_path}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.results_path, index=False)
        print(f"Saved quantile evaluation results → {self.results_path}")
        return results_df


if __name__ == '__main__':
    qr = QuantileRegression()
    res = qr.run()
    print(res)
