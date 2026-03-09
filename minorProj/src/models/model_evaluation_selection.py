"""
Step 27: Model Evaluation & Selection

Aggregates and compares all model evaluation results (baseline, linear regression,
LightGBM, quantile, multi-output) side by side, ranks them, and selects the best model.
"""

import pandas as pd
from pathlib import Path

class ModelEvaluationSelection:
    def __init__(self, processed_dir='data/processed'):
        self.processed_dir = Path(processed_dir)
        self.summary_path = self.processed_dir / 'model_comparison_summary.csv'
        self.best_model_path = self.processed_dir / 'best_model.txt'

    def load_results(self):
        results = []

        # Baseline results
        baseline_path = self.processed_dir / 'baseline_results.csv'
        if baseline_path.exists():
            df = pd.read_csv(baseline_path)
            df['source'] = 'baseline'
            results.append(df)

        # Linear Regression results
        lr_path = self.processed_dir / 'linear_regression_results.csv'
        if lr_path.exists():
            df = pd.read_csv(lr_path)
            df['source'] = 'linear_regression'
            results.append(df)

        # LightGBM results
        lgb_path = self.processed_dir / 'lightgbm_results.csv'
        if lgb_path.exists():
            df = pd.read_csv(lgb_path)
            df['source'] = 'lightgbm'
            results.append(df)

        # Quantile results (median alpha=0.5)
        qr_path = self.processed_dir / 'quantile_results.csv'
        if qr_path.exists():
            df = pd.read_csv(qr_path)
            # Take the median quantile (alpha=0.5) for comparison
            df_median = df[df['alpha'] == 0.5].copy()
            if not df_median.empty:
                df_median['model'] = 'Quantile (alpha=0.5)'
                df_median['source'] = 'quantile'
                df_median = df_median.rename(columns={'pinball_loss': 'pinball_loss_0.5'})
                results.append(df_median[['model', 'mae', 'source']])

        # Multi-output results (total_orders target)
        mo_path = self.processed_dir / 'multioutput_results.csv'
        if mo_path.exists():
            df = pd.read_csv(mo_path)
            df_orders = df[df['target'] == 'total_orders'].copy()
            if not df_orders.empty:
                df_orders['model'] = 'MultiOutput (orders)'
                df_orders['source'] = 'multioutput'
                results.append(df_orders[['model', 'rmse', 'mae', 'source']])

        return results

    def run(self):
        all_results = self.load_results()
        if not all_results:
            print('No model results found.')
            return None

        combined = pd.concat(all_results, ignore_index=True)

        # Normalize columns
        if 'rmse' not in combined.columns:
            combined['rmse'] = None
        if 'mae' not in combined.columns:
            combined['mae'] = None
        if 'r2' not in combined.columns:
            combined['r2'] = None

        combined = combined[['model', 'source', 'rmse', 'mae', 'r2']].copy()

        # Sort by RMSE (ascending), then MAE
        combined_sorted = combined.sort_values(by=['rmse', 'mae'], ascending=[True, True], na_position='last')

        # Save summary
        combined_sorted.to_csv(self.summary_path, index=False)
        print(f'Model comparison summary saved → {self.summary_path}\n')
        print(combined_sorted.to_string(index=False))

        # Best model selection (lowest RMSE)
        best_row = combined_sorted.dropna(subset=['rmse']).iloc[0]
        best_model = best_row['model']
        best_rmse = best_row['rmse']
        best_mae = best_row['mae']

        with open(self.best_model_path, 'w') as f:
            f.write(f'Best Model: {best_model}\n')
            f.write(f'RMSE: {best_rmse}\n')
            f.write(f'MAE: {best_mae}\n')

        print(f'\n✅ Best Model: {best_model} (RMSE={best_rmse:.4f}, MAE={best_mae:.4f})')
        print(f'Saved best model info → {self.best_model_path}')

        return combined_sorted

if __name__ == '__main__':
    evaluator = ModelEvaluationSelection()
    evaluator.run()
