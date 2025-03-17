import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset
df = pd.read_csv("/Users/marie/Documents/GitHub/YB-2407-DA/fittness_tracking/dataset.csv")

# Peek at dataset
print("-" * 120)
print("First 5 rows:")
print(df.head())
print("\nShape of the dataset:", df.shape)

print("-" * 120)
print("Info:")
print(df.info())

# Basic summary stats
print("-" * 120)
print("Descriptive Statistics:")
print(df.describe(include="all"))

# Check for missing values
print("-" * 120)
print("Missing Values:")
print(df.isnull().sum())

# Remove improbable outliers

# Select numeric columns
numeric_columns = df.select_dtypes(include=['number']).columns

# Dictionary to hold outlier indices for each column
outlier_indices = {}

# Loop through each numeric column to detect outliers using the IQR method
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify indices of outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_indices[col] = outliers.index.tolist()

    if len(outliers) > 0:
        plt.boxplot(df[col])
        plt.title(f"Box Plot for '{col}'")
        plt.ylabel('Values')
        plt.show()

        print("-" * 120)
        print(f"Column '{col}' has {len(outliers)} outliers.")

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print("Outliers using IQR method:")
        print(outliers)

        # Remove rows with outlier values
        cleaned_data = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        print(f"Cleaned dataset shape '{col}': {cleaned_data.shape}")

# Combine all outlier indices
print("-" * 120)
all_outliers = set(idx for indices in outlier_indices.values() for idx in indices)
print(f"Total unique outlier records: {len(all_outliers)}")

print("-" * 120)
