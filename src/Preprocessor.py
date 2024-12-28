import pandas as pd
import numpy as np
import argparse
import os

# Function to handle missing values
def handle_missing_values(df, threshold=0.5):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        missing_ratio = df[col].isnull().sum() / len(df)
        if missing_ratio > threshold:
            df.drop(col, axis=1, inplace=True)
        else:
            df[col] = df[col].fillna(df[col].mean())

    return df

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='C:/Users/ibsan/Desktop/TenX/week-3/Data/raw/MachineLearningRating_v3.csv')
    parser.add_argument('--output', required=True, help='C:/Users/ibsan/Desktop/TenX/week-3/Data/preprocessed')
    args = parser.parse_args()

    data_path = args.input
    output_path = args.output

    # Load the data
    data = pd.read_csv(data_path, low_memory=False)

    # Exclude columns 32 and 37 (zero-based indices 31 and 36)
    columns_to_exclude = [31, 36]
    data = data.drop(data.columns[columns_to_exclude], axis=1)

    # Handle missing values in numerical columns
    data = handle_missing_values(data)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the cleaned data
    data.to_csv(output_path, index=False)
