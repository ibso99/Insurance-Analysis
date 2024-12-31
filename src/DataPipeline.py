import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import os
import logging

class DataPreprocessor:
    def __init__(self, output_dir="preprocessed_data", test_size=0.2, random_state=42, config=None):
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.config = config if config else {}
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up basic logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    class FeatureEngineer(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X = X.copy()
            if 'RegistrationYear' in X.columns:
                X['VehicleAge'] = pd.to_datetime('now').year - X['RegistrationYear']
                X.drop('RegistrationYear', axis=1, inplace=True)
                logging.info("Feature 'VehicleAge' created from 'RegistrationYear'.")

            if 'VehicleIntroDate' in X.columns:
                X['VehicleIntroYear'] = pd.to_datetime(X['VehicleIntroDate'], format='%Y-%m-%d', errors='coerce').dt.year
                X['VehicleIntroYear'] = X['VehicleIntroYear'].fillna(X['VehicleIntroYear'].median())
                X.drop('VehicleIntroDate', axis=1, inplace=True)
                logging.info("Feature 'VehicleIntroYear' created from 'VehicleIntroDate'.")
            
            return X

    def process_data(self, df, target_premium='TotalPremium', target_claims='TotalClaims'):
        # Handle missing values in the dataframe
        if df.isnull().sum().sum() > 0:
            logging.warning(f"Data contains missing values. Handling missing data...")

        self.df = df.copy()

        # Remove problematic columns 32 and 37
        if len(self.df.columns) > 37:
            columns_to_remove = [self.df.columns[32], self.df.columns[37]]
            self.df = self.df.drop(columns=columns_to_remove)
            logging.info(f"Removed columns: {columns_to_remove}")
        else:
            logging.warning("The dataframe does not have enough columns to remove 32 and 37.")

        # Apply feature engineering
        engineer = self.FeatureEngineer()
        self.df = engineer.transform(self.df)

        # Convert column dtypes
        self._convert_column_dtypes(self.df)

        # Separate features and targets
        X = self.df.drop([target_premium, target_claims], axis=1, errors='ignore')
        y_premium = self.df[target_premium] if target_premium in self.df.columns else None
        y_claims = self.df[target_claims] if target_claims in self.df.columns else None

        # Impute missing target values
        if y_premium is not None:
            y_premium.fillna(y_premium.median(), inplace=True)
        if y_claims is not None:
            y_claims.fillna(y_claims.median(), inplace=True)

        # Identify numerical and categorical columns
        numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
        logging.info(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical columns.")

        # Create preprocessing pipeline
        preprocessor = self._create_preprocessing_pipeline(numerical_cols, categorical_cols)

        # Apply preprocessing steps
        X_processed = preprocessor.fit_transform(X)
        X_processed = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

        # Combine features and targets for saving
        processed_data = X_processed.copy()
        if y_premium is not None:
            processed_data[target_premium] = y_premium.values
        if y_claims is not None:
            processed_data[target_claims] = y_claims.values

        # Save processed data to CSV
        self._save_processed_data(processed_data)

        # Split data into training and testing sets
        X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
            X_processed, y_premium, test_size=self.test_size, random_state=self.random_state
        ) if y_premium is not None else (None, None, None, None)

        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_processed, y_claims, test_size=self.test_size, random_state=self.random_state
        ) if y_claims is not None else (None, None, None, None)

        return X_train_p, X_test_p, y_train_p, y_test_p, X_train_c, X_test_c, y_train_c, y_test_c

    def _create_preprocessing_pipeline(self, numerical_cols, categorical_cols):
        numerical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )

        return preprocessor

    def _save_processed_data(self, processed_data):
        output_file = os.path.join(self.output_dir, 'processed_data_with_targets.csv')
        processed_data.to_csv(output_file, index=False)
        logging.info(f"Processed data with targets saved to {output_file}")

    def _convert_column_dtypes(self, df):
        def convert_column_dtype(col):
            if df[col].dtype == object:
                df[col] = df[col].replace({',': '.'}, regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif df[col].dtype in [int, float]:
                if df[col].apply(lambda x: isinstance(x, float)).any():
                    df[col] = df[col].astype(float)

        for col in df.columns:
            convert_column_dtype(col)
