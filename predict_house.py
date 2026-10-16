import pandas as pd
import joblib

# 1. Load the saved model
model = joblib.load("house_price_model.pkl")

# 2. Load the dataset
df = pd.read_csv("data/train.csv")

# 3. Select one house for testing
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

house = X.iloc[[0]]
actual_price = y.iloc[0]

# 4. Predict the house price
predicted_price = model.predict(house)[0]

# 5. Display results
print("\n--- House Price Prediction Test ---")
print("Actual Price:", round(actual_price, 2))
print("Predicted Price:", round(predicted_price, 2))
print("Absolute Error:", round(abs(actual_price - predicted_price), 2))
