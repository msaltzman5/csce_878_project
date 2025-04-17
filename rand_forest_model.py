import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("data/cab_rides.csv")

# Drop rows with missing target values
df = df.dropna(subset=["price"])

# Drop unnecessary columns
df = df.drop(columns=["id", "product_id"])  # keep product_id for now

# One-hot encode categorical columns manually
categorical_cols = ["cab_type", "destination", "source", "name"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(df.head())

# Define features and target
X = df_encoded.drop("price", axis=1)
y = df_encoded["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rfModel = RandomForestRegressor(n_estimators=100, random_state=42)
rfModel.fit(X_train, y_train)

# Predict and evaluate
y_pred = rfModel.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

results = X_test.copy()
results["actual_price"] = y_test
results["predicted_price"] = y_pred

print(rfModel.score(X, np.ravel(y)))

# Display a few sample outputs
print(results[["actual_price", "predicted_price"]])

# Optional: Predict a sample
sample = X.iloc[0:1]
predicted_price = rfModel.predict(sample)
print(f"Predicted price for sample 0: ${predicted_price[0]:.2f}")
