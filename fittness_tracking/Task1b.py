import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('/Users/marie/Documents/GitHub/YB-2407-DA/fittness_tracking/dataset.csv')

# Convert categorical columns to 'category' type and then encode them
df["Gender"] = df["Gender"].astype("category")
df["Activity Level"] = df["Activity Level"].astype("category")
df["Location"] = df["Location"].astype("category")

df["Gender_Code"] = df["Gender"].cat.codes          # Male=0, Female=1 (alphabetical by default)
df["Activity_Code"] = df["Activity Level"].cat.codes  # Active=0, Moderate=1, Sedentary=2 (alphabetical)
df["Location_Code"] = df["Location"].cat.codes        # Rural=0, Suburban=1, Urban=2 (alphabetical)

# Define predictor features and the target variable
# Here, we use demographic and physical activity metrics as predictors
features = ["Age", "Gender_Code", "Activity_Code", "Location_Code", 
            "Distance Travelled (km)", "Calories Burned"]
target = "App Sessions"

X = df[features]
y = df[target]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)

# Visualize the actual vs. predicted App Sessions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.8, edgecolors='b')
plt.xlabel("Actual App Sessions")
plt.ylabel("Predicted App Sessions")
plt.title("Actual vs. Predicted App Sessions")
# Plot a reference line for perfect prediction
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'red', lw=2)
plt.tight_layout()
plt.show()