# compare_with_sklearn.py
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error, r2_score

from simpleLinearRegression import SimpleLinearRegression  


# Load real dataset
data = load_diabetes()
X = data.data[:, 2]    # BMI feature
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# Train Scratch Model
# ===============================
scratch_model = SimpleLinearRegression()
scratch_model.fit(X_train, y_train)

scratch_pred = scratch_model.predict(X_test)
scratch_mse = scratch_model.MSE(X_test, y_test)
scratch_r2 = scratch_model.score(X_test, y_test)
scratch_adj_r2 = scratch_model.adj_r2(X_test, y_test)

# ===============================
# Train sklearn Model
# ===============================
sk_model = SklearnLR()
sk_model.fit(X_train.reshape(-1, 1), y_train)

sk_pred = sk_model.predict(X_test.reshape(-1, 1))
sk_mse = mean_squared_error(y_test, sk_pred)
sk_r2 = r2_score(y_test, sk_pred)

# Adjusted R² for sklearn
n = len(y_test)
p = 1
sk_adj_r2 = 1 - (1 - sk_r2) * (n - 1) / (n - p - 1)

# ===============================
# Output Comparison
# ===============================
print("\n===== PARAMETER COMPARISON =====")
print(f"Scratch slope     : {scratch_model.slope}")
print(f"Sklearn slope     : {sk_model.coef_[0]}")
print()
print(f"Scratch intercept : {scratch_model.intercept}")
print(f"Sklearn intercept : {sk_model.intercept_}")

print("\n===== METRIC COMPARISON =====")
print(f"Scratch MSE       : {scratch_mse}")
print(f"Sklearn MSE       : {sk_mse}")
print()
print(f"Scratch R²        : {scratch_r2}")
print(f"Sklearn R²        : {sk_r2}")
print()
print(f"Scratch Adj R²    : {scratch_adj_r2}")
print(f"Sklearn Adj R²    : {sk_adj_r2}")

print("\n===== VERDICT =====")
print("R² match :", abs(scratch_r2 - sk_r2) < 1e-6)
print("MSE match:", abs(scratch_mse - sk_mse) < 1e-6)
