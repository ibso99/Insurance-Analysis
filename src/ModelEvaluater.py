import os
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class ModelEvaluator:
    def __init__(self, models, X, y, test_size=0.2, random_state=42, output_dir="model_metrics"):
        self.models = models
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure the directory exists

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def evaluate(self):
        results = []
        for name, model in self.models.items():
            logging.info(f"Training model: {name}")
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)

            # Calculate metrics
            mae = mean_absolute_error(self.y_test, predictions)
            mse = mean_squared_error(self.y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, predictions)

            # Log metrics
            logging.info(f"Model: {name} - MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")

            # Save metrics to results list
            results.append({
                "Model": name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2
            })

        # Save the results to a CSV file
        results_df = pd.DataFrame(results)
        metrics_file = os.path.join(self.output_dir, "evaluation_metrics.csv")
        results_df.to_csv(metrics_file, index=False)
        logging.info(f"Metrics saved to {metrics_file}")


