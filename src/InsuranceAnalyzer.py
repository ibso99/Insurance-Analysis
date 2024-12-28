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
          # Load data with explicit dtype and low_memory=False
            self.df = pd.read_csv(dataframe_path, low_memory=False)
            self.df = self.df.copy() # Create a copy here instead of later
            print("Data loaded and copied successfully.")

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
            if missing_percentage > 0 :
              if pd.api.types.is_numeric_dtype(self.df[col]):
                   try:
                       # Convert to numeric, handling commas as decimal
                       self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', '.'), errors='raise')

                       if missing_percentage < 0.5:
                             mean_value = self.df[col].mean()
                             self.df[col].fillna(mean_value, inplace=True)
                             print(f"Filled missing values in {col} with mean.")

                       elif missing_percentage >=0.5 :
                           self.df.drop(col, axis=1, inplace=True)
                           print(f"Removed column {col} due to excessive missing values.")
                   except ValueError as e:
                       print(f"Error converting column '{col}' to numeric: {e}")
                       self.df.drop(col, axis=1, inplace=True) # Drop the column if conversion fails
                       print(f"Dropping column '{col}'")

              else:
                   print(f"Column {col} is not numeric, skipping imputation and conversion.")

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
        # Filter numerical cols
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]
        print("\nDescriptive Statistics for numerical features:")
        print(self.df[numerical_cols].describe())

        print("\nData Structure (dtypes):")
        print(self.df.dtypes)


    def univariate_analysis(self):
        
        numerical_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured',
                          'RegistrationYear','NumberOfDoors','cubiccapacity','kilowatts','CustomValueEstimate',
                          'CapitalOutstanding','NumberOfVehiclesInFleet']
        
        categorical_cols = ['IsVATRegistered', 'Citizenship', 'LegalType', 'Title', 'Language', 'Bank', 'AccountType',
                          'MaritalStatus', 'Gender', 'Country', 'Province', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone',
                          'ItemType', 'VehicleType', 'make', 'Model', 'bodytype','AlarmImmobiliser', 'TrackingDevice','NewVehicle', 
                          'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder', 'CoverCategory', 'CoverType',
                          'CoverGroup', 'Section', 'Product', 'StatutoryClass', 'StatutoryRiskType','TermFrequency','ExcessSelected']
        
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]
         # Plot histograms for numerical columns in a grid
        num_numerical = len(numerical_cols)
        num_rows = (num_numerical + 1) // 2  # Calculate rows needed for 2 plots per row

        fig, axes = plt.subplots(num_rows, 2, figsize=(16, num_rows * 5))
        axes = axes.flatten()  # Flatten the axes array for easy indexing
        for i, col in enumerate(numerical_cols):
            sns.histplot(self.df[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')

        # Remove empty subplots if any
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        # Plot bar charts for categorical columns in a grid
        num_categorical = len(categorical_cols)
        num_rows = (num_categorical + 1) // 2  # Calculate rows needed for 2 plots per row

        fig, axes = plt.subplots(num_rows, 2, figsize=(16, num_rows * 5))
        axes = axes.flatten()  # Flatten the axes array for easy indexing
        for i, col in enumerate(categorical_cols):
            if self.df[col].nunique() <= 20:  # Avoid plotting too many categories
              sns.countplot(data=self.df, y=col, ax=axes[i])
              axes[i].set_title(f'Distribution of {col}')
              axes[i].tick_params(axis='x', rotation=45)
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()


    def bivariate_analysis(self):
         print("Analysing correlations between TotalPremium and TotalClaims with different features")
         # Correlation matrix for numerical features
         numerical_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm',
                            'SumInsured','RegistrationYear','NumberOfDoors','cubiccapacity',
                            'kilowatts','CustomValueEstimate','CapitalOutstanding','NumberOfVehiclesInFleet']
         numerical_cols = [col for col in numerical_cols if col in self.df.columns]

         # Ensure all columns are converted to numeric before calculating correlation
         for col in numerical_cols:
           try:
               # Convert to numeric, handling commas as decimal
              self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', '.'), errors='raise')
           except ValueError as e:
              print(f"Error converting column '{col}' to numeric in Bivariate analysis: {e}")
              self.df.drop(col, axis=1, inplace=True)
              print(f"Dropping column '{col}' in Bivariate analysis")
              numerical_cols.remove(col)


         corr_matrix = self.df[numerical_cols].corr()
         plt.figure(figsize=(12, 8))
         sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
         plt.title("Correlation Matrix of Numerical Features")
         plt.show()
         selected_cat_features = ['Gender', 'Province', 'VehicleType', 'CoverCategory']
        # Scatter plots of TotalPremium and TotalClaims against selected features in a grid
         num_features = len(selected_cat_features)
         num_axes = num_features * 2  # Correct allocation of axes to take 2 plots each
         num_rows = (num_axes + 1) // 2  # Calculate rows needed for 2 plots per row
         fig, axes = plt.subplots(num_rows, 2, figsize=(16, num_rows * 6))
         axes = axes.flatten()

         for i, cat_feature in enumerate(selected_cat_features):
             if 2*i < len(axes) :
                sns.scatterplot(x=cat_feature, y='TotalPremium', data=self.df, ax=axes[2*i])
                axes[2*i].set_title(f'TotalPremium vs {cat_feature}')
                axes[2*i].tick_params(axis='x', rotation=45)
             if 2*i + 1 < len(axes):
                sns.scatterplot(x=cat_feature, y='TotalClaims', data=self.df, ax=axes[2*i+1])
                axes[2*i+1].set_title(f'TotalClaims vs {cat_feature}')
                axes[2*i+1].tick_params(axis='x', rotation=45)
         # Remove empty subplots if any
         for j in range(2*i+2, len(axes)):
           fig.delaxes(axes[j])
         plt.tight_layout()
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
               fig, axes = plt.subplots(1, 2, figsize=(16, 5))

               sns.barplot(data=grouped, x='CoverType', y='TotalPremium', ax=axes[0])
               axes[0].set_title(f'Average Premium for Cover Types in {province}')
               axes[0].tick_params(axis='x', rotation=45)

               sns.barplot(data=grouped, x='CoverType', y='TotalClaims', ax=axes[1])
               axes[1].set_title(f'Average Claims for Cover Types in {province}')
               axes[1].tick_params(axis='x', rotation=45)
               plt.tight_layout()
               plt.show()

                 # Auto make
            make_grouped = province_df.groupby('make')[['TotalPremium']].mean().reset_index()
            if not make_grouped.empty:
                plt.figure(figsize=(16, 6))
                sns.barplot(data=make_grouped, x='make', y='TotalPremium')
                plt.title(f'Average Premium for Auto makes in {province}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout() 
                plt.show()


    def outlier_detection(self):
        """
        Detects and visualizes outliers using box plots for selected numerical features in a grid format.
        """
        numerical_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 
                          'SumInsured','RegistrationYear','NumberOfDoors','cubiccapacity',
                          'kilowatts','CustomValueEstimate','CapitalOutstanding','NumberOfVehiclesInFleet']
        
        numerical_cols = [col for col in numerical_cols if col in self.df.columns]
        num_features = len(numerical_cols)
        num_rows = (num_features + 1) // 2  # Calculate rows needed for 2 plots per row
        fig, axes = plt.subplots(num_rows, 2, figsize=(16, num_rows * 5))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.boxplot(x=self.df[col], ax=axes[i])
            axes[i].set_title(f'Boxplot for Outlier Detection in {col}')

         # Remove empty subplots if any
        for j in range(i+1, len(axes)):
           fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

    def visualization(self):
      """
      Produces 3 creative plots that summarize insights from the EDA in a grid format
      """
      # 1. Pairplot of selected numerical features
      print("Generating a pairplot for selected numerical features..")
      numerical_cols_vis = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'SumInsured']
      numerical_cols_vis = [col for col in numerical_cols_vis if col in self.df.columns]
      sns.pairplot(self.df[numerical_cols_vis], diag_kind='kde')
      plt.suptitle("Pairplot of Selected Numerical Features", y=1.02)
      plt.show()


      # 2. Distribution of Premium and claims by Province
      print("Generating distribution of premium and claim based on provinces")
      fig, axes = plt.subplots(1,2,figsize=(16,8))

      sns.violinplot(data=self.df, x='Province', y='TotalPremium', inner='quartile', ax = axes[0])
      axes[0].set_title("Distribution of Total Premium by Province")
      axes[0].tick_params(axis='x', rotation=45)
      ax = axes[0]


      sns.violinplot(data=self.df, x='Province', y='TotalClaims', inner='quartile',ax=axes[1])
      axes[1].set_title("Distribution of Total Claims by Province")
      axes[1].tick_params(axis='x', rotation=45)
      ax = axes[1]
      plt.tight_layout()
      plt.show()


      # 3. Count Plot of Vehicle Type by Marital Status
      print("Generating countplot of vehicle type by marital status..")
      plt.figure(figsize=(12, 8))
      sns.countplot(data=self.df, x='VehicleType', hue='MaritalStatus')
      plt.title("Distribution of Vehicle Type by Marital Status")
      plt.xticks(rotation=45)
      plt.show()