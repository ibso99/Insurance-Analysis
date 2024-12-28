import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class InsuranceDataAnalyzer:
    """
    A class for performing Exploratory Data Analysis (EDA) on insurance data.
    """

    def __init__(self, dataframe_path="data.csv"):
        try:
            self.df = pd.read_csv(dataframe_path)
            print("Data loaded successfully")
        except FileNotFoundError:
            print(f"Error: File '{dataframe_path}' not found. Please provide correct path")
            exit()


    def data_quality_assessment(self):
        print("\nMissing Values per Column:")
        print(self.df.isnull().sum())

    def handle_missing_values(self):
         """
          Fills missing values with the mean if less than 50% of the column is missing and the column is of numeric type,
           otherwise, it removes the column. Returns the cleaned dataframe.
          """
         print("\nHandling missing values...")
         for col in self.df.columns:
            missing_percentage = self.df[col].isnull().sum() / len(self.df)
            if missing_percentage > 0:
              if missing_percentage < 0.5 and pd.api.types.is_numeric_dtype(self.df[col]):
                 mean_value = self.df[col].mean()
                 self.df[col].fillna(mean_value, inplace=True)
                 print(f"Filled missing values in {col} with mean.")
              elif missing_percentage >=0.5 :
                  self.df.drop(col, axis=1, inplace=True)
                  print(f"Removed column {col} due to excessive missing values.")
              else:
                print(f"Column {col} is not numeric, skipping imputation.")

         print("Missing values handled.")
         return self.df


    def data_summarization(self):
        """
        Calculates and displays descriptive statistics of numerical features,
        and reviews the data types for each column.
        """
        numerical_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured',
                          'RegistrationYear','NumberOfDoors','cubiccapacity','kilowatts','CustomValueEstimate',
                          'CapitalOutstanding','NumberOfVehiclesInFleet']
        print("\nDescriptive Statistics for numerical features:")
        print(self.df[numerical_cols].describe())

        print("\nData Structure (dtypes):")
        print(self.df.dtypes)



    def univariate_analysis(self):
        """
        Generates and displays histograms for numerical columns
        and bar plots for categorical columns.
        """
        numerical_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured',
                          'RegistrationYear','NumberOfDoors','cubiccapacity','kilowatts','CustomValueEstimate',
                          'CapitalOutstanding','NumberOfVehiclesInFleet']
        
        categorical_cols = ['IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 'AccountType',
                          'MaritalStatus', 'Gender', 'Country', 'Province', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone',
                          'ItemType', 'VehicleType', 'make', 'Model', 'bodytype','AlarmImmobiliser', 'TrackingDevice',
                          'NewVehicle', 'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'CoverCategory', 'CoverType',
                          'CoverGroup', 'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType','TermFrequency','ExcessSelected']

        # Plot histograms for numerical columns
        for col in numerical_cols:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

        # Plot bar charts for categorical columns
        for col in categorical_cols:
            if self.df[col].nunique() <= 20:  # Avoid plotting too many categories
                plt.figure(figsize=(10, 6))
                sns.countplot(data=self.df, y=col)
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45)
                plt.show()

    def bivariate_analysis(self):
        """
        Explore relationships between TotalPremium, TotalClaims and other features.
        """
        print("Analysing correlations between TotalPremium and TotalClaims with different features")

        # Correlation matrix for numerical features

        numerical_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured',
                          'RegistrationYear','NumberOfDoors','cubiccapacity','kilowatts','CustomValueEstimate',
                          'CapitalOutstanding','NumberOfVehiclesInFleet']
        
        corr_matrix = self.df[numerical_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Matrix of Numerical Features")
        plt.show()

        # Scatter plots of TotalPremium and TotalClaims against selected features
        selected_cat_features = ['Gender', 'Province', 'VehicleType', 'CoverCategory']
        for cat_feature in selected_cat_features:
            plt.figure(figsize=(12, 6))
            sns.scatterplot(x=cat_feature, y='TotalPremium', data=self.df)
            plt.title(f'TotalPremium vs {cat_feature}')
            plt.xticks(rotation=45)
            plt.show()

            plt.figure(figsize=(12, 6))
            sns.scatterplot(x=cat_feature, y='TotalClaims', data=self.df)
            plt.title(f'TotalClaims vs {cat_feature}')
            plt.xticks(rotation=45)
            plt.show()

    def data_comparison(self):
        print("\nComparing trends based on  insurance cover type, premium and auto make..")
        # Compare changes in premium and cover type across provinces
        provinces = self.df['Province'].unique()

        if 'Province' in self.df.columns:
            for province in provinces:
                province_df = self.df[self.df['Province'] == province]
                # Group by cover type and calculate mean premium and claims
                grouped = province_df.groupby('CoverType')[['TotalPremium', 'TotalClaims']].mean().reset_index()
                if not grouped.empty:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=grouped, x='CoverType', y='TotalPremium')
                    plt.title(f'Average Premium for Cover Types in {province}')
                    plt.xticks(rotation=45)
                    plt.show()

                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=grouped, x='CoverType', y='TotalClaims')
                    plt.title(f'Average Claims for Cover Types in {province}')
                    plt.xticks(rotation=45)
                    plt.show()


                # Auto make
                make_grouped = province_df.groupby('make')[['TotalPremium']].mean().reset_index()
                if not make_grouped.empty:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=make_grouped, x='make', y='TotalPremium')
                    plt.title(f'Average Premium for Auto makes in {province}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
                    plt.show()

    def outlier_detection(self):
        numerical_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 
                          'SumInsured','RegistrationYear','NumberOfDoors','cubiccapacity',
                          'kilowatts','CustomValueEstimate','CapitalOutstanding','NumberOfVehiclesInFleet']

        for col in numerical_cols:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot for Outlier Detection in {col}')
            plt.show()

    def visualization(self):
        """
        Produces 3 creative plots that summarize insights from the EDA.
        """
        # 1. Pairplot of selected numerical features
        print("Generating a pairplot for selected numerical features..")
        numerical_cols_vis = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured']
        sns.pairplot(self.df[numerical_cols_vis], diag_kind='kde')
        plt.suptitle("Pairplot of Selected Numerical Features", y=1.02)
        plt.show()

        # 2. Distribution of Premium and claims by Province
        print("Generating distribution of premium and claim based on provinces")
        plt.figure(figsize=(12, 8))
        sns.violinplot(data=self.df, x='Province', y='TotalPremium', inner='quartile')
        plt.title("Distribution of Total Premium by Province")
        plt.xticks(rotation=45, ha='right')
        plt.show()

        plt.figure(figsize=(12, 8))
        sns.violinplot(data=self.df, x='Province', y='TotalClaims', inner='quartile')
        plt.title("Distribution of Total Claims by Province")
        plt.xticks(rotation=45, ha='right')
        plt.show()

        # 3. Count Plot of Vehicle Type by Marital Status
        print("Generating countplot of vehicle type by marital status..")
        plt.figure(figsize=(12, 8))
        sns.countplot(data=self.df, x='VehicleType', hue='MaritalStatus')
        plt.title("Distribution of Vehicle Type by Marital Status")
        plt.xticks(rotation=45)
        plt.show()