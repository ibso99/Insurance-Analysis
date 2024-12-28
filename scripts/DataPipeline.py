import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataPipeline:
    
    def __init__(self, data):
        self.data = data

    def data_structure(self):
        print(self.data.info())

    def missing_values(self):
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])

    def impute_missing_values(self):
        # Drop columns with more than 60% missing values
        threshold = 0.6
        self.data = self.data.loc[:, self.data.isnull().mean() < threshold]
        
        # Impute missing values with mean for numerical columns
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            self.data[col].fillna(self.data[col].mean())

    def data_summary(self):
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        print(self.data[numerical_cols].describe())

    def univariate_analysis(self):
        num_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        cat_cols = self.data.select_dtypes(include=['object']).columns
        
        num_plots = len(num_cols)
        cat_plots = len(cat_cols)
        total_plots = num_plots + cat_plots
        
        fig, axes = plt.subplots(total_plots, 1, figsize=(10, 5 * total_plots))
        
        for i, col in enumerate(num_cols):
            self.data[col].hist(bins=30, ax=axes[i])
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].set_title(f'Distribution of {col}')
        
        for i, col in enumerate(cat_cols):
            self.data[col].value_counts().plot(kind='bar', ax=axes[num_plots + i])
            axes[num_plots + i].set_xlabel(col)
            axes[num_plots + i].set_ylabel('Count')
            axes[num_plots + i].set_title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.show()

    def bivariate_analysis(self):
        # Scatter plot
        plt.figure(figsize=(8, 4))
        plt.scatter(self.data['TotalPremium'], self.data['TotalClaims'])
        plt.xlabel('Total Premium')
        plt.ylabel('Total Claims')
        plt.title('Relationship between Total Premium and Total Claims')
        plt.show()

        # Correlation matrix
        corr_matrix = self.data[['TotalPremium', 'TotalClaims', 'SumInsured']].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def data_comparison(self):
        # Compare TotalPremium by ZipCode
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='ZipCode', y='TotalPremium', data=self.data)
        plt.title('Total Premium by ZipCode')
        plt.xticks(rotation=90)
        plt.show()

    def outlier_detection(self):
        # detecting outliers using box plots
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.data[col])
            plt.title(f'Box plot of {col}')
            plt.show()

    def advanced_visualization(self):
        #   Pair plot
        sns.pairplot(self.data[['TotalPremium', 'TotalClaims', 'SumInsured']])
        plt.title('Pair Plot')
        plt.show()

        # Violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='ZipCode', y='TotalClaims', data=self.data)
        plt.title('Violin Plot of Total Claims by ZipCode')
        plt.xticks(rotation=90)
        plt.show()

        #  Heatmap of correlations
        corr_matrix = self.data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Heatmap of Correlations')
        plt.show()