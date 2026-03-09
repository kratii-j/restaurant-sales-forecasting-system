"""
Step 28: Restaurant-Specific Analysis

Analyzes prediction errors and model performance at the restaurant level.
Outputs per-restaurant RMSE, MAE, and identifies best/worst performing restaurants.
"""

import numpy as np
import pandas as pd
from pathlib import Path

class RestaurantSpecificAnalysis:
    def __init__(self, predictions_path='data/processed/multioutput_predictions.csv'):
        self.predictions_path = Path(predictions_path)
        self.output_dir = Path('data/processed')
        self.per_restaurant_path = self.output_dir / 'restaurant_performance.csv'
        self.summary_path = self.output_dir / 'restaurant_analysis_summary.csv'

    def load_predictions(self):
        df = pd.read_csv(self.predictions_path)
        return df

    def compute_metrics(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        return rmse, mae

    def run(self):
        df = self.load_predictions()

        # Ensure columns exist
        if 'pred_total_orders' not in df.columns or 'total_orders' not in df.columns:
            print('Missing required columns in predictions file.')
            return None

        # Group by restaurant_id
        grouped = df.groupby('restaurant_id')
        results = []

        for rid, grp in grouped:
            y_true = grp['total_orders'].values
            y_pred = grp['pred_total_orders'].values
            rmse, mae = self.compute_metrics(y_true, y_pred)
            results.append({
                'restaurant_id': rid,
                'n_samples': len(grp),
                'rmse': float(rmse),
                'mae': float(mae),
                'mean_actual': float(y_true.mean()),
                'mean_predicted': float(y_pred.mean()),
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('rmse', ascending=True).reset_index(drop=True)

        # Save per-restaurant performance
        results_df.to_csv(self.per_restaurant_path, index=False)
        print(f'Per-restaurant performance saved → {self.per_restaurant_path}')

        # Summary: best and worst 5 restaurants
        best5 = results_df.head(5)
        worst5 = results_df.tail(5)

        summary = pd.concat([
            best5.assign(category='best'),
            worst5.assign(category='worst')
        ])
        summary.to_csv(self.summary_path, index=False)
        print(f'Restaurant analysis summary saved → {self.summary_path}\n')

        print('===== Top 5 Best Performing Restaurants =====')
        print(best5.to_string(index=False))
        print('\n===== Top 5 Worst Performing Restaurants =====')
        print(worst5.to_string(index=False))

        # Overall stats
        print('\n===== Overall Restaurant Stats =====')
        print(f'Total restaurants: {len(results_df)}')
        print(f'Mean RMSE across restaurants: {results_df["rmse"].mean():.4f}')
        print(f'Median RMSE across restaurants: {results_df["rmse"].median():.4f}')
        print(f'Std RMSE across restaurants: {results_df["rmse"].std():.4f}')

        return results_df

if __name__ == '__main__':
    analysis = RestaurantSpecificAnalysis()
    analysis.run()
