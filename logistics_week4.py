
"""
Week 4 Project: Predictive Modeling and Optimization
in Logistics Systems

Objective:
Predict actual shipping days using machine learning
and suggest logistics optimization strategies.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 1. Generate hypothetical logistics dataset

rng = np.random.default_rng(42)
n = 500

shipping_mode = rng.choice(
    ["Standard Class", "Second Class",
     "First Class", "Same Day"],
    n,
    p=[0.45, 0.28, 0.20, 0.07]
)

region = rng.choice(
    ["Central", "East", "South", "West"], n
)

scheduled = rng.integers(2, 8, n)
quantity = rng.integers(1, 15, n)
sales = np.round(rng.uniform(50, 1500, n), 2)

mode_effect = pd.Series(shipping_mode).map({
    "Standard Class": 1.3,
    "Second Class": 0.7,
    "First Class": 0.2,
    "Same Day": -0.7
}).to_numpy()

region_effect = pd.Series(region).map({
    "Central": 0.2,
    "East": 0.0,
    "South": 0.5,
    "West": 0.3
}).to_numpy()

noise = rng.normal(0, 0.8, n)

actual_days = np.maximum(
    1,
    scheduled + mode_effect + region_effect
    + 0.035 * quantity - 0.00025 * sales + noise
)

data = pd.DataFrame({
    "Shipping_Mode": shipping_mode,
    "Order_Region": region,
    "Shipping_Days_Scheduled": scheduled,
    "Order_Quantity": quantity,
    "Sales": sales,
    "Shipping_Days_Actual": np.round(actual_days, 2)
})

# 2. Separate features and target

X = data.drop(columns=["Shipping_Days_Actual"])
y = data["Shipping_Days_Actual"]

categorical = ["Shipping_Mode", "Order_Region"]

numerical = [
    "Shipping_Days_Scheduled",
    "Order_Quantity",
    "Sales"
]

# 3. Data preprocessing

def build_preprocessor():
    return ColumnTransformer([
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical
        )
    ])

# 4. Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# 5. Select machine learning models

models = {
    "Linear Regression": Pipeline([
        ("prep", build_preprocessor()),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            random_state=42
        ))
    ])
}

# 6. Train and evaluate models

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    cv_mae = -cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_mean_absolute_error"
    ).mean()

    results[name] = mae

    print("\nModel:", name)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))
    print("Cross Validation MAE:", round(cv_mae, 3))

# 7. Select the best model

best_name = min(results, key=results.get)
best_model = models[best_name]

print("\nBest Model:", best_name)

# 8. Predict delivery time for a new order

sample_order = pd.DataFrame([{
    "Shipping_Mode": "Standard Class",
    "Order_Region": "West",
    "Shipping_Days_Scheduled": 5,
    "Order_Quantity": 4,
    "Sales": 500.0
}])

prediction = best_model.predict(sample_order)

print(
    "Predicted Delivery Days:",
    round(float(prediction[0]), 2)
)

# 9. Logistics optimization recommendations

print("\nOPTIMIZATION RECOMMENDATIONS")

print("1. Identify potentially delayed shipments.")
print("2. Improve delivery route planning.")
print("3. Balance workload across delivery teams.")
print("4. Monitor high-priority shipments.")
print("5. Continuously evaluate model performance.")

print("\nWeek 4 Project Completed Successfully!")
