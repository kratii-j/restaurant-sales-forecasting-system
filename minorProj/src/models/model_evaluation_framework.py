import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ModelEvaluationFramework:
    def __init__(self, predictions_path, true_values_path, results_path):
        self.predictions_path = Path(predictions_path)
        self.true_values_path = Path(true_values_path)
        self.results_path = Path(results_path)

    def load_data(self):
        y_pred = pd.read_csv(self.predictions_path)
        y_true = pd.read_csv(self.true_values_path)
        return y_true, y_pred

    def evaluate(self, y_true, y_pred):
        # Assume y_true and y_pred are aligned and have 'total_orders' column
        y_true = y_true['total_orders']
        y_pred = y_pred['total_orders']
        mse = mean_squared_error(y_true, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    def run(self):
        y_true, y_pred = self.load_data()
        metrics = self.evaluate(y_true, y_pred)
        results_df = pd.DataFrame([metrics])
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(self.results_path, index=False)
        print(f"Evaluation results saved → {self.results_path}\n")
        print(results_df)

if __name__ == "__main__":
    # Example usage for linear regression
    framework = ModelEvaluationFramework(
        predictions_path="data/processed/test.csv",  # Use model predictions here
        true_values_path="data/processed/test.csv",  # Use true values here
        results_path="data/processed/model_evaluation_results.csv"
    )
    framework.run()
