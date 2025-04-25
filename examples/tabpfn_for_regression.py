#  Copyright (c) Prior Labs GmbH 2025.
"""Example of using TabPFN for regression.

This example demonstrates how to use TabPFNRegressor on a regression task
using the diabetes dataset from scikit-learn.
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#from src.tabpfn.regressor import TabPFNRegressor
from tabpfn import TabPFNRegressor

# Load data
X, y = load_diabetes(return_X_y=True)
"""X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)"""

# Use numpy Generator for reproducibility
rng = np.random.default_rng(42)
n_samples = 200  # Reduce the number of samples
random_indices = rng.choice(X.shape[0], n_samples, replace=False)
X_subsampled = X[random_indices]
y_subsampled = y[random_indices]

X_train, X_test, y_train, y_test = train_test_split(
    X_subsampled,
    y_subsampled,
    test_size=0.33,
    random_state=42,
)


# Initialize a regressor
reg = TabPFNRegressor()
reg.fit(X_train, y_train)

# Predict a point estimate (using the mean)
predictions = reg.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# Predict quantiles
quantiles = [0.25, 0.5, 0.75]
quantile_predictions = reg.predict(
    X_test,
    output_type="quantiles",
    quantiles=quantiles,
)
for q, q_pred in zip(quantiles, quantile_predictions):
    print(f"Quantile {q} MAE:", mean_absolute_error(y_test, q_pred))
# Predict with mode
mode_predictions = reg.predict(X_test, output_type="mode")
print("Mode MAE:", mean_absolute_error(y_test, mode_predictions))
