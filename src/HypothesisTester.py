import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import logging
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InsuranceABTester:
    def __init__(self, df):
        self.df = df.copy()
        self._clean_data()
        self._calculate_profit()

    def _clean_data(self):
        """Performs data cleaning and preprocessing."""

        # Drop columns with > 50% missing values
        missing_percentage = (self.df.isnull().sum() / len(self.df)) * 100
        cols_to_drop = missing_percentage[missing_percentage > 50].index
        self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        logging.info(f"Dropped columns with >50% missing values: {list(cols_to_drop)}")

        # Drop other irrelevant columns
        cols_to_drop = ['NumberOfVehiclesInFleet']  # This was always empty
        self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        logging.info(f"Dropped irrelevant columns: {cols_to_drop}")

        # Handle remaining missing values
        logging.info("Handling missing values...")
        for col in self.df.columns:
            missing_percentage = self.df[col].isnull().sum() / len(self.df)
            if missing_percentage > 0:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    try:
                        self.df[col] = pd.to_numeric(
                            self.df[col].astype(str).str.replace(',', '', regex=False), errors='coerce'
                        )

                        if missing_percentage < 0.5:
                            mean_value = self.df[col].mean()
                            self.df[col].fillna(mean_value, inplace=True)
                            logging.info(f"Filled missing values in {col} with mean: {mean_value}")
                        else:
                            self.df.drop(col, axis=1, inplace=True)
                            logging.warning(f"Removed column {col} due to excessive missing values.")
                    except ValueError as e:
                        logging.error(f"Error converting column '{col}' to numeric: {e}")
                        self.df.drop(col, axis=1, inplace=True)
                else:
                    mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                    self.df[col].fillna(mode_value, inplace=True)
                    logging.info(f"Filled missing values in non-numeric column {col} with mode: {mode_value}")
        logging.info("Missing values handled.")

        # Convert to datetime
        date_cols = ['TransactionMonth', 'VehicleIntroDate']
        for col in date_cols:
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                logging.info(f"Converted column {col} to datetime.")
            except (ValueError, KeyError):
                logging.warning(f"Date conversion failed for column: {col}. Check column names and format.")

        # Boolean Conversion
        bool_cols = ['IsVATRegistered', 'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted']
        for col in bool_cols:
            try:
                self.df[col] = (
                    self.df[col]
                    .map({'Yes': True, 'No': False, True: True, False: False})
                    .fillna(False)
                    .astype(bool)
                )
                logging.info(f"Converted column {col} to boolean.")
            except KeyError:
                logging.warning(f"Boolean conversion failed for column: {col}. Check column names and values.")

        # Categorical Conversion
        cat_cols = ['Gender', 'MaritalStatus', 'AlarmImmobiliser', 'TrackingDevice', 'TermFrequency', 'ExcessSelected']
        for col in cat_cols:
            try:
                self.df[col] = self.df[col].astype('category')
                logging.info(f"Converted column {col} to categorical.")
            except KeyError:
                logging.warning(f"Category conversion failed for column: {col}. Check column names.")

    def _calculate_profit(self):
        if {'TotalPremium', 'TotalClaims'}.issubset(self.df.columns):
            self.df['Profit'] = self.df['TotalPremium'] - self.df['TotalClaims']
            logging.info("Calculated Profit column.")
        else:
            logging.warning("One or both of 'TotalPremium' and 'TotalClaims' are missing. Skipping profit calculation.")

    def perform_ab_testing(self):
        self._test_hypothesis_anova("Province", "TotalClaims", "risk")
        self._test_hypothesis_anova("PostalCode", "TotalClaims", "risk")
        self._test_hypothesis_anova("PostalCode", "Profit", "profit")
        self._test_gender_hypothesis_anova()

    def _test_hypothesis_anova(self, group_col, metric_col, metric_type):
        if group_col not in self.df.columns or metric_col not in self.df.columns:
            logging.warning(f"Missing column(s) for hypothesis testing: {group_col}, {metric_col}")
            return

        # Create formula for linear regression
        formula = f"{metric_col} ~ C({group_col})"

        # Fit the model
        model = ols(formula, data=self.df).fit()

        # Perform ANOVA test
        anova_table = anova_lm(model, typ=2)

        # Print ANOVA table
        print(anova_table)
        
        # Interpret of the results 
        if anova_table["PR(>F)"].iloc[0] <= 0.05:
            logging.info(f"Reject null hypothesis: Significant {metric_type} difference found.")
        else:
            logging.info(f"Fail to reject null hypothesis: No significant {metric_type} difference found.")

    
    def _test_gender_hypothesis_anova(self):
        if 'Gender' not in self.df.columns or 'TotalClaims' not in self.df.columns:
            logging.warning("Missing columns for gender hypothesis testing.")
            return

        # Create formula for linear regression with categorical gender
        formula = f"TotalClaims ~ C(Gender)"

        # Fit the model
        model = ols(formula, data=self.df).fit()

        # Perform ANOVA test
        anova_table = anova_lm(model, typ=2)

        # Print ANOVA table
        print(anova_table)

        #   Interpret of the results 
        if anova_table["PR(>F)"].iloc[0] <= 0.05:
            logging.info(f"Reject null hypothesis: Significant risk difference found.")
        else:
            logging.info(f"Fail to reject null hypothesis: No significant risk difference found.")