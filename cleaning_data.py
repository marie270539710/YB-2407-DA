import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (replace 'path_to_file.csv' with the actual file path)
file_path = 'iris/iris.data' 
data = pd.read_csv(file_path, header=None)

# Assign appropriate column names
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data.columns = column_names

# Data cleaning: Check for and remove missing values
cleaned_data = data.dropna()

# Convert numerical columns to float
for col in cleaned_data.columns[:-1]:  # Exclude 'species'
    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')

# Display the first 5 rows of the cleaned dataset
print("Top 5 Data:\n\n",cleaned_data.head())

# Calculate the correlation matrix for numerical columns
correlation_matrix = data.iloc[:, :-1].corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Heatmap")
plt.show()
