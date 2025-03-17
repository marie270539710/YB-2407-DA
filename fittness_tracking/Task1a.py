import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1.1 Load dataset
df = pd.read_csv("/Users/marie/Documents/GitHub/YB-2407-DA/fittness_tracking/dataset.csv")

# 1.2 Peek at dataset
print("First 5 rows:")
print(df.head())

print("\nShape of the dataset:", df.shape)
print("\nInfo:")
print(df.info())

# 1.3 Basic summary stats
print("\nDescriptive Statistics:")
print(df.describe(include="all"))

# 1.4 Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())
