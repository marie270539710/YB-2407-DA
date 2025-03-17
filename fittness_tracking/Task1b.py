import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('/Users/marie/Documents/GitHub/YB-2407-DA/fittness_tracking/dataset.csv')

# Convert categorical columns to 'category' type and then encode them
df["Gender"] = df["Gender"].astype("category")
df["Activity Level"] = df["Activity Level"].astype("category")
df["Location"] = df["Location"].astype("category")

df["Gender_Code"] = df["Gender"].cat.codes          # Male=0, Female=1 (alphabetical by default)
df["Activity_Code"] = df["Activity Level"].cat.codes  # Active=0, Moderate=1, Sedentary=2 (alphabetical)
df["Location_Code"] = df["Location"].cat.codes        # Rural=0, Suburban=1, Urban=2 (alphabetical)


###############################################################
###############     Regression Analysis         ###############
###############################################################


# Define predictor features and the target variable
# Here, we use demographic and physical activity metrics as predictors
features = ["Age", "Gender_Code", "Activity_Code", "Location_Code", 
            "Distance Travelled (km)", "Calories Burned"]
target = "App Sessions"

X = df[features]
y = df[target]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("-" * 120)
print("Model Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)
print("-" * 120)

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


###############################################################
###############     Classification Models       ###############
###############################################################

# Create a binary target for classification: High Engagement (1) vs. Low Engagement (0)
# Here, we arbitrarily define "High Engagement" as users having >= 100 app sessions
df["High_Engagement"] = np.where(df["App Sessions"] >= 100, 1, 0)

# Define features (predictors) and the target
features = ["Age", "Gender_Code", "Activity_Code", "Location_Code", 
            "Distance Travelled (km)", "Calories Burned"]
X = df[features]
y = df["High_Engagement"]

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Initialize and train the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict on the test set
y_pred_log = log_reg.predict(X_test)

# Evaluate the model
print("-" * 120)
print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# Initialize and train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("-" * 120)
print("Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("-" * 120)