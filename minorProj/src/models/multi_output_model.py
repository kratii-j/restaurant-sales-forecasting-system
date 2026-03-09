import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


class MultiOutputModel:
    def __init__(self, train_path='data/processed/train.csv', test_path='data/processed/test.csv'):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.results_path = Path('data/processed/multi_output_results.csv')
        self.predictions_path = Path('data/processed/multi_output_predictions.csv')
        self.model_path = Path('models/multi_output_model.joblib')

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return train, test

    def prepare_features(self, df):
        # Drop identifier/time and targets
        drop_cols = ['restaurant_id', 'date', 'total_orders', 'total_revenue']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        # Fill numeric NaNs with median
        X = X.fillna(X.median(numeric_only=True))
        return X

    def run(self):
        train, test = self.load_data()
        X_train = self.prepare_features(train)
        X_test = self.prepare_features(test)

        y_train = train[['total_orders', 'total_revenue']].values
        y_test_df = test[['total_orders', 'total_revenue']].copy()
        y_test = y_test_df.values

        # Use RandomForest wrapped in MultiOutputRegressor as a robust baseline
        base = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model = MultiOutputRegressor(base)

        print('Training multi-output model...')
        model.fit(X_train, y_train)

        print('Predicting test set...')
        y_pred = model.predict(X_test)

        # Evaluation per target
        results = []
        target_names = ['total_orders', 'total_revenue']
        for i, name in enumerate(target_names):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            results.append({'target': name, 'rmse': float(rmse), 'mae': float(mae)})

        # Save results
        results_df = pd.DataFrame(results)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.results_path, index=False)

        # Save predictions (append to test dataframe)
        pred_df = test.copy()
        pred_df['pred_total_orders'] = y_pred[:, 0]
        pred_df['pred_total_revenue'] = y_pred[:, 1]
        pred_df.to_csv(self.predictions_path, index=False)

        # Save model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

        print(f'Multi-output model saved → {self.model_path}')
        print(f'Predictions saved → {self.predictions_path}')
        print(f'Evaluation saved → {self.results_path}')
        print(results_df)

        return results_df


if __name__ == '__main__':
    MultiOutputModel().run()
