import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

class MultiOutputModel:
    def __init__(self, train_path='data/processed/train.csv', test_path='data/processed/test.csv'):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.predictions_path = Path('data/processed/multioutput_predictions.csv')
        self.results_path = Path('data/processed/multioutput_results.csv')
        self.best_params_path = Path('data/processed/lightgbm_best_params.csv')

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return train, test

    def prepare_features(self, df):
        drop_cols = ['restaurant_id', 'date', 'total_orders', 'total_revenue']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        # Fill numeric NaNs with median
        X = X.fillna(X.median(numeric_only=True))
        return X

    def _load_best_params(self):
        if self.best_params_path.exists():
            try:
                bp = pd.read_csv(self.best_params_path).iloc[0].to_dict()
                # coerce numeric types
                params = {}
                for k, v in bp.items():
                    try:
                        v2 = int(v)
                    except Exception:
                        try:
                            v2 = float(v)
                        except Exception:
                            v2 = v
                    params[k] = v2
                return params
            except Exception:
                return {}
        return {}

    def evaluate_pair(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae

    def run(self):
        train, test = self.load_data()
        X_train = self.prepare_features(train)
        X_test = self.prepare_features(test)

        y_train = train[['total_orders', 'total_revenue']]
        y_test = test[['total_orders', 'total_revenue']]

        best_params = self._load_best_params()
        # default LightGBM params (safe)
        lgb_params = {
            'n_estimators': int(best_params.get('n_estimators', 200)),
            'learning_rate': float(best_params.get('learning_rate', 0.1)),
            'num_leaves': int(best_params.get('num_leaves', 50)),
            'max_depth': int(best_params.get('max_depth', -1)),
            'random_state': int(best_params.get('random_state', 42)),
        }

        # Validate numeric params
        try:
            if lgb_params['learning_rate'] <= 0:
                print('Invalid learning_rate in best params; resetting to 0.1')
                lgb_params['learning_rate'] = 0.1
        except Exception:
            lgb_params['learning_rate'] = 0.1

        print('Using LightGBM params:', lgb_params)

        base_estimator = lgb.LGBMRegressor(**lgb_params)
        model = MultiOutputRegressor(base_estimator)

        print('Training multi-output model (orders + revenue) ...')
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        preds_df = pd.DataFrame(preds, columns=['pred_total_orders', 'pred_total_revenue'])

        # Save predictions merged with test keys
        out_df = test.reset_index(drop=True).copy()
        out_df = pd.concat([out_df, preds_df], axis=1)
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(self.predictions_path, index=False)
        print(f'Saved multi-output predictions → {self.predictions_path}')

        # Evaluate per-target
        rmse_orders, mae_orders = self.evaluate_pair(y_test['total_orders'], preds_df['pred_total_orders'])
        rmse_revenue, mae_revenue = self.evaluate_pair(y_test['total_revenue'], preds_df['pred_total_revenue'])

        results = pd.DataFrame([
            {'target': 'total_orders', 'rmse': float(rmse_orders), 'mae': float(mae_orders)},
            {'target': 'total_revenue', 'rmse': float(rmse_revenue), 'mae': float(mae_revenue)},
            {'target': 'aggregate_mean', 'rmse': float(np.mean([rmse_orders, rmse_revenue])), 'mae': float(np.mean([mae_orders, mae_revenue]))}
        ])

        results.to_csv(self.results_path, index=False)
        print(f'Saved multi-output evaluation results → {self.results_path}')
        print(results)
        return out_df, results

if __name__ == '__main__':
    m = MultiOutputModel()
    m.run()
