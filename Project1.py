import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb
import shap
import numpy as np

# Set a global random seed for reproducibility
np.random.seed(42)

# Define models and their scores
models = ['Random Forest', 'Decision Tree', 'XGBoost', 'Gradient Boosting']
r2_scores = [0.98, 0.97, 0.97, 0.94]
mse_scores = [0.02, 0.03, 0.03, 0.06]

# Plot R2 Scores
plt.figure(figsize=(10, 5))
plt.bar(models, r2_scores, color='blue')
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('R2 Scores of Regression Models')
plt.ylim(0.9,1.0)
plt.savefig("R2 Score Comparison.png")
plt.show()

# Plot MSE Scores
plt.figure(figsize=(10, 5))
plt.bar(models, mse_scores, color='red')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.title('MSE of Regression Models')
plt.savefig("MSE Score Comparison.png")
plt.show()

# Load data
try:
    data = pd.read_csv('supply_logistics_Final_1.csv')
except FileNotFoundError:
    print("The file 'supply_logistics_Final_1.csv' was not found.")
    # Handle error or exit

# Feature selection and preprocessing
features = ['Product ID', 'Unit Quantity', 'Destination Port']
X = pd.get_dummies(data[features])
y_total_cost = data['Total Cost']
y_origin_port = data['Origin Port']
y_carrier = data['Carrier']

# Encode categorical target variables
le_port = LabelEncoder()
le_carrier = LabelEncoder()

y_origin_port_encoded = le_port.fit_transform(y_origin_port)
y_carrier_encoded = le_carrier.fit_transform(y_carrier)

# Split data into training and test sets with a fixed random state
X_train, X_test, y_cost_train, y_cost_test = train_test_split(X, y_total_cost, test_size=0.2, random_state=42)
_, _, y_port_train, y_port_test = train_test_split(X, y_origin_port_encoded, test_size=0.2, random_state=42)
_, _, y_carrier_train, y_carrier_test = train_test_split(X, y_carrier_encoded, test_size=0.2, random_state=42)

# Initialize models with random states
models_regression = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

models_classification = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train models for Total Cost prediction (regression)
for name, model in models_regression.items():
    model.fit(X_train, y_cost_train)

# Train models for Origin Port prediction (classification)
for name, model in models_classification.items():
    model.fit(X_train, y_port_train)

# Train models for Carrier prediction (classification)
for name, model in models_classification.items():
    model.fit(X_train, y_carrier_train)

for i in range(0,3):
    # User input for prediction
    product_id = int(input("Enter Product ID: "))
    unit_quantity = int(input("Enter Unit Quantity: "))
    destination_port = input("Enter Destination Port: ")

    # Prepare input for prediction
    input_data = pd.DataFrame([[product_id, unit_quantity, destination_port]], columns=features)
    input_data_encoded = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)

    # Ensure all data is of type float64
    X_train = X_train.astype(float)
    input_data_encoded = input_data_encoded.astype(float)

    # Predict using Random Forest as an example
    total_cost_prediction_rf = models_regression['Random Forest'].predict(input_data_encoded)[0]
    origin_port_prediction_rf = models_classification['Random Forest'].predict(input_data_encoded)[0]
    carrier_prediction_rf = models_classification['Random Forest'].predict(input_data_encoded)[0]

    print(f"Predicted Total Cost (RF): {total_cost_prediction_rf}")
    print(f"Predicted Origin Port (RF): {le_port.inverse_transform([origin_port_prediction_rf])[0]}")
    print(f"Predicted Carrier (RF): {le_carrier.inverse_transform([carrier_prediction_rf])[0]}")

# Set random seed for reproducibility
np.random.seed(0)

# Read dataset back from CSV file
df = pd.read_csv("synthetic_data.csv")

# --- 1. SHAP Summary Plot for Carrier-related Prediction ---
# Using all 6 features to predict Total Cost for Carrier-related prediction
X_carrier = df[["Total Cost", "Unit Quantity", "Cost per unit", "Product ID", "Origin Port", "Destination Port"]]
y_carrier = df["Weight"]  # We predict the Weight as a proxy for Carrier-related info

# Split into training and testing sets
X_train_carrier, X_test_carrier, y_train_carrier, y_test_carrier = train_test_split(X_carrier, y_carrier, test_size=0.2, random_state=0)

# Train an XGBoost model
model_carrier = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, random_state=0)
model_carrier.fit(X_train_carrier, y_train_carrier)

# SHAP analysis for Carrier
explainer_carrier = shap.Explainer(model_carrier, X_train_carrier)
shap_values_carrier = explainer_carrier(X_test_carrier)

# Generate SHAP summary plot for Carrier
shap.summary_plot(shap_values_carrier, X_test_carrier, feature_names=X_carrier.columns)
plt.title('SHAP Summary Plot for Carrier')
plt.show()

# --- 2. SHAP Summary Plot for Total Cost Prediction ---
# Using all features to predict Total Cost
X_total_cost = df.drop(columns=["Total Cost"])
y_total_cost = df["Total Cost"]

# Split into training and testing sets
X_train_total_cost, X_test_total_cost, y_train_total_cost, y_test_total_cost = train_test_split(X_total_cost, y_total_cost, test_size=0.2, random_state=0)

# Train an XGBoost model
model_total_cost = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, random_state=0)
model_total_cost.fit(X_train_total_cost, y_train_total_cost)

# SHAP analysis for Total Cost
explainer_total_cost = shap.Explainer(model_total_cost, X_train_total_cost)
shap_values_total_cost = explainer_total_cost(X_test_total_cost)

# Generate SHAP summary plot for Total Cost
shap.summary_plot(shap_values_total_cost, X_test_total_cost, feature_names=X_total_cost.columns)
plt.title('SHAP Summary Plot for Total Cost')
plt.show()

# --- 3. SHAP Summary Plot for Origin Port Prediction ---
# Using all features except Origin Port to predict Origin Port
X_origin_port = df.drop(columns=["Origin Port"])
y_origin_port = df["Origin Port"]

# Split into training and testing sets
X_train_origin_port, X_test_origin_port, y_train_origin_port, y_test_origin_port = train_test_split(X_origin_port, y_origin_port, test_size=0.2, random_state=0)

# Train an XGBoost model
model_origin_port = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=5, random_state=0)
model_origin_port.fit(X_train_origin_port, y_train_origin_port)

# SHAP analysis for Origin Port
explainer_origin_port = shap.Explainer(model_origin_port, X_train_origin_port)
shap_values_origin_port = explainer_origin_port(X_test_origin_port)

# Generate SHAP summary plot for Origin Port
shap.summary_plot(shap_values_origin_port, X_test_origin_port, feature_names=X_origin_port.columns)
plt.title('SHAP Summary Plot for Origin Port')
plt.show()
