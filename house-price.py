
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
df = pd.read_csv("data/train.csv")

print("\n--- Dataset Overview ---")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\n--- Dataset Information ---")
df.info()

print("\n--- Statistical Summary ---")
print(df.describe())

# 2. Missing Values Analysis
print("\n--- Missing Values ---")

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_summary = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percent
})

missing_summary = missing_summary[
    missing_summary["Missing_Count"] > 0
].sort_values(by="Missing_Percentage", ascending=False)

print(missing_summary.round(2))

# 3. Check Duplicate Rows
print("\n--- Duplicate Rows ---")
print(df.duplicated().sum())

# 4. Exploratory Data Analysis (EDA)

# Graph 1: Distribution of House Prices
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="SalePrice", kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")
plt.tight_layout()
plt.show()

# Graph 2: Overall Quality vs Sale Price
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="OverallQual", y="SalePrice")
plt.title("Overall Quality vs Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.tight_layout()
plt.show()

# Graph 3: Living Area vs Sale Price
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice")
plt.title("Living Area vs Sale Price")
plt.xlabel("Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.tight_layout()
plt.show()

# 5. Separate Features and Target
X = df.drop(columns=["SalePrice", "Id"])
y = df["SalePrice"]

numeric_cols = X.select_dtypes(include="number").columns
categorical_cols = X.select_dtypes(exclude="number").columns

print("\n--- Features and Target ---")
print("Features:", X.shape)
print("Target:", y.shape)
print("Numerical Columns:", len(numeric_cols))
print("Categorical Columns:", len(categorical_cols))

# 6. Preprocessing Pipelines

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Missing"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# 7. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n--- Train-Test Split ---")
print("Training Rows:", X_train.shape[0])
print("Testing Rows:", X_test.shape[0])

# 8. Define Models

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}

# 9. Train and Evaluate Models

results = {}
trained_pipelines = {}

for name, model in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    print(f"\n--- {name} ---")

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    }

    trained_pipelines[name] = pipeline

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

# 10. Compare Model Performance

results_df = pd.DataFrame(results).T

print("\n--- Model Comparison ---")
print(results_df.round(4))

# 11. Select Random Forest and Predict

best_model = trained_pipelines["Random Forest"]
rf_predictions = best_model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Price": y_test.to_numpy(),
    "Predicted Price": rf_predictions
})

comparison["Error"] = (
    comparison["Actual Price"] - comparison["Predicted Price"]
)

comparison["Absolute Error"] = comparison["Error"].abs()

print("\n--- Actual vs Predicted: First 10 Houses ---")
print(comparison.head(10).round(2))

print(
    "\nMean Absolute Error:",
    round(comparison["Absolute Error"].mean(), 2)
)

# 12. Actual vs Predicted Plot: RED DOTS

plt.figure(figsize=(8, 6))

plt.scatter(
    comparison["Actual Price"],
    comparison["Predicted Price"],
    color="red",
    s=35,
    alpha=0.7,
    label="Predictions"
)

min_price = min(
    comparison["Actual Price"].min(),
    comparison["Predicted Price"].min()
)

max_price = max(
    comparison["Actual Price"].max(),
    comparison["Predicted Price"].max()
)

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    color="black",
    linestyle="--",
    label="Ideal Prediction"
)

plt.title("Random Forest: Actual vs Predicted Prices")
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.legend()
plt.tight_layout()
plt.show()

# 13. Residual Plot

plt.figure(figsize=(8, 5))

plt.scatter(
    comparison["Predicted Price"],
    comparison["Error"],
    color="red",
    s=35,
    alpha=0.7
)

plt.axhline(
    y=0,
    color="black",
    linestyle="--",
    label="Zero Error"
)

plt.title("Random Forest: Prediction Errors")
plt.xlabel("Predicted Sale Price")
plt.ylabel("Actual Price - Predicted Price")
plt.legend()
plt.tight_layout()
plt.show()

# 14. Save Trained Model

joblib.dump(best_model, "house_price_model.pkl")

print("\nRandom Forest model saved successfully!")
print("Saved file: house_price_model.pkl")

print("\nModel training, evaluation and saving completed!")

