import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

###############################################################
###############         Data Cleaning           ###############
###############################################################

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
        print(f"{col} has {len(outliers)} outliers.")

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

# Confirming Associated Metrics
plt.figure(figsize=(10,6))
plt.scatter(df['App Sessions'], df['Distance Travelled (km)'], color='blue', label="Distance Travelled (km)")
plt.scatter(df['App Sessions'], df['Calories Burned'], color='red', label="Calories Burned")
plt.xlabel("App Sessions")
plt.ylabel("Metrics")
plt.title("App Sessions vs. Distance Travelled and Calories Burned")
plt.legend()
plt.show()


###############################################################
############### Exploratory Data Analysis (EDA) ###############
###############################################################


# Convert columns to 'category' type
df["Gender"] = df["Gender"].astype("category")
df["Activity Level"] = df["Activity Level"].astype("category")
df["Location"] = df["Location"].astype("category")

# Now encode them into numeric codes
df["Gender"] = df["Gender"].cat.codes          # Male=0, Female=1 (alphabetical by default)
df["Activity Level"] = df["Activity Level"].cat.codes  # Active=0, Moderate=1, Sedentary=2 (alphabetical)
df["Location"] = df["Location"].cat.codes        # Rural=0, Suburban=1, Urban=2 (alphabetical)

# Compute correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='viridis', interpolation='none')
plt.colorbar()
plt.title('Fitness Tracking: Correlation Matrix')
plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
plt.tight_layout()
plt.show()