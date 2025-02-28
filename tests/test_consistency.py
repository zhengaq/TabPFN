"""Model consistency tests for TabPFN.

These tests verify that TabPFN models produce consistent predictions across code
changes. If predictions change, developers must explicitly acknowledge and verify
the improvement.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

# mypy: ignore-errors
from tabpfn import TabPFNClassifier, TabPFNRegressor  # type: ignore


def get_prediction_hash(model_predictions: np.ndarray) -> str:
    """Generate a deterministic hash from model predictions.

    Args:
        model_predictions: Array of model predictions to hash

    Returns:
        A deterministic string hash of the predictions
    """
    # Convert to fixed precision string format first for deterministic hashing
    prediction_str = np.array2string(
        model_predictions,
        precision=8,
        suppress_small=True,
    )
    return hashlib.sha256(prediction_str.encode("utf-8")).hexdigest()


# Define reference hashes for expected prediction outputs
# These should only be updated when model improvements are verified
REFERENCE_HASHES = {
    "iris_classifier": (
        "74d8ebd26158c13936819d378da6dd1bd3d776336327db44d7065ff5e9a1a305"
    ),
    "breast_cancer_classifier": (
        "2afcc05426156fde62f7c1d8d5a7c34916b0fafd0f5849dd4f57ce2da13e2b68"
    ),
    "boston_regressor": (
        "e173b38a048088ca34ac095ddf2fc237a34426c76b4c2729afa453cdd27ffaaa"
    ),
    "diabetes_regressor": (
        "280c25c599e6d16bf13b4e3398a9b10bb7a8161004d5df2209ead2f30bfe8ed0"
    ),
}


class TestModelConsistency:
    """Verify that TabPFN models produce consistent predictions across code changes."""

    @pytest.fixture
    def iris_data(self):
        """Fixture for Iris dataset."""
        # Use fixed random state for reproducibility
        random_state = check_random_state(42)
        X, y = sklearn.datasets.load_iris(return_X_y=True, as_frame=False)

        # Standardize features for deterministic results
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Shuffle with fixed random state
        indices = np.arange(len(X))
        random_state.shuffle(indices)
        X, y = X[indices], y[indices]

        # Split into train/test
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def breast_cancer_data(self):
        """Fixture for Breast Cancer dataset."""
        # Use fixed random state for reproducibility
        random_state = check_random_state(42)
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=False)

        # Standardize features for deterministic results
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Shuffle with fixed random state
        indices = np.arange(len(X))
        random_state.shuffle(indices)
        X, y = X[indices], y[indices]

        # Split into train/test
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def boston_data(self):
        """Fixture for Boston Housing dataset."""
        # Use fixed random state for reproducibility
        random_state = check_random_state(42)

        # Boston dataset is deprecated but we can load it from sklearn-datasets
        # or use California housing as an alternative
        try:
            # Try to use fetch_openml to get Boston dataset
            from sklearn.datasets import fetch_openml

            housing = fetch_openml(data_id=531, as_frame=False)
            X, y = housing.data, housing.target
        except (ImportError, ValueError):
            # Fall back to California housing
            from sklearn.datasets import fetch_california_housing

            housing = fetch_california_housing(as_frame=False)
            X, y = housing.data, housing.target

        # Standardize features for deterministic results
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Shuffle with fixed random state
        indices = np.arange(len(X))
        random_state.shuffle(indices)
        X, y = X[indices], y[indices]

        # Split into train/test
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    @pytest.fixture
    def diabetes_data(self):
        """Fixture for Diabetes dataset."""
        # Use fixed random state for reproducibility
        random_state = check_random_state(42)
        X, y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=False)

        # Standardize features for deterministic results
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Shuffle with fixed random state
        indices = np.arange(len(X))
        random_state.shuffle(indices)
        X, y = X[indices], y[indices]

        # Split into train/test
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    def test_iris_classifier_consistency(self, iris_data):
        """Verify TabPFNClassifier predictions on Iris dataset are consistent."""
        X_train, X_test, y_train, y_test = iris_data

        # Create classifier with fixed random state
        clf = TabPFNClassifier(
            n_estimators=2,  # Low number for faster tests
            random_state=42,
            device="cpu",  # Use CPU for deterministic results
        )

        # Fit and predict
        clf.fit(X_train, y_train)
        predictions = clf.predict_proba(X_test)

        # Generate hash of predictions
        prediction_hash = get_prediction_hash(predictions)

        # Verify hash matches expected value
        assert prediction_hash == REFERENCE_HASHES["iris_classifier"], (
            f"TabPFNClassifier predictions for Iris have changed.\n"
            f"Expected hash: {REFERENCE_HASHES['iris_classifier']}\n"
            f"Actual hash: {prediction_hash}\n\n"
            f"If this change is intentional:\n"
            f"1. Verify the changes improve model performance on benchmarks\n"
            f"2. Update the reference hash in `tests/test_consistency.py`\n"
            f"3. Document the improvement in your PR description\n"
        )

    def test_breast_cancer_classifier_consistency(self, breast_cancer_data):
        """Verify TabPFNClassifier predictions on Breast Cancer dataset."""
        X_train, X_test, y_train, y_test = breast_cancer_data

        # Create classifier with fixed random state
        clf = TabPFNClassifier(
            n_estimators=2,  # Low number for faster tests
            random_state=42,
            device="cpu",  # Use CPU for deterministic results
        )

        # Fit and predict
        clf.fit(X_train, y_train)
        predictions = clf.predict_proba(X_test)

        # Generate hash of predictions
        prediction_hash = get_prediction_hash(predictions)

        # Verify hash matches expected value
        assert prediction_hash == REFERENCE_HASHES["breast_cancer_classifier"], (
            f"TabPFNClassifier predictions for Breast Cancer have changed.\n"
            f"Expected hash: {REFERENCE_HASHES['breast_cancer_classifier']}\n"
            f"Actual hash: {prediction_hash}\n\n"
            f"If this change is intentional:\n"
            f"1. Verify the changes improve model performance on benchmarks\n"
            f"2. Update the reference hash in `tests/test_consistency.py`\n"
            f"3. Document the improvement in your PR description\n"
        )

    def test_boston_regressor_consistency(self, boston_data):
        """Verify TabPFNRegressor predictions on Boston Housing dataset."""
        X_train, X_test, y_train, y_test = boston_data

        # Create regressor with fixed random state
        reg = TabPFNRegressor(
            n_estimators=2,  # Low number for faster tests
            random_state=42,
            device="cpu",  # Use CPU for deterministic results
        )

        # Fit and predict
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)

        # Generate hash of predictions
        prediction_hash = get_prediction_hash(predictions)

        # Verify hash matches expected value
        assert prediction_hash == REFERENCE_HASHES["boston_regressor"], (
            f"TabPFNRegressor predictions for Boston Housing have changed.\n"
            f"Expected hash: {REFERENCE_HASHES['boston_regressor']}\n"
            f"Actual hash: {prediction_hash}\n\n"
            f"If this change is intentional:\n"
            f"1. Verify the changes improve model performance on benchmarks\n"
            f"2. Update the reference hash in `tests/test_consistency.py`\n"
            f"3. Document the improvement in your PR description\n"
        )

    def test_diabetes_regressor_consistency(self, diabetes_data):
        """Verify TabPFNRegressor predictions on Diabetes dataset are consistent."""
        X_train, X_test, y_train, y_test = diabetes_data

        # Create regressor with fixed random state
        reg = TabPFNRegressor(
            n_estimators=2,  # Low number for faster tests
            random_state=42,
            device="cpu",  # Use CPU for deterministic results
        )

        # Fit and predict
        reg.fit(X_train, y_train)
        predictions = reg.predict(X_test)

        # Generate hash of predictions
        prediction_hash = get_prediction_hash(predictions)

        # Verify hash matches expected value
        assert prediction_hash == REFERENCE_HASHES["diabetes_regressor"], (
            f"TabPFNRegressor predictions for Diabetes have changed.\n"
            f"Expected hash: {REFERENCE_HASHES['diabetes_regressor']}\n"
            f"Actual hash: {prediction_hash}\n\n"
            f"If this change is intentional:\n"
            f"1. Verify the changes improve model performance on benchmarks\n"
            f"2. Update the reference hash in `tests/test_consistency.py`\n"
            f"3. Document the improvement in your PR description\n"
        )


class TestHashRobustness:
    """Verify that our hash function correctly detects changes."""

    @pytest.fixture
    def test_data(self):
        """Create a simple dataset for testing."""
        check_random_state(42)
        X, y = sklearn.datasets.make_classification(
            n_samples=100,
            n_features=10,
            random_state=42,
        )

        # Split into train/test
        train_size = 70
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        return X_train, X_test, y_train, y_test

    def test_classifier_hash_detects_data_changes(self, test_data):
        """Verify hash function detects changes in classification input data."""
        X_train, X_test, y_train, y_test = test_data

        # Create classifier with fixed settings
        clf = TabPFNClassifier(n_estimators=2, random_state=42, device="cpu")
        clf.fit(X_train, y_train)

        # Get predictions on original data
        original_predictions = clf.predict_proba(X_test)
        original_hash = get_prediction_hash(original_predictions)

        # Modify test data slightly
        X_test_modified = X_test.copy()
        X_test_modified[0, 0] += 0.1  # Small change to first feature

        # Get predictions on modified data
        modified_predictions = clf.predict_proba(X_test_modified)
        modified_hash = get_prediction_hash(modified_predictions)

        # Verify hash detects the change
        assert (
            original_hash != modified_hash
        ), "Hash function failed to detect change in classification data"

    def test_classifier_hash_detects_model_changes(self, test_data):
        """Verify hash function detects changes in classifier configuration."""
        X_train, X_test, y_train, y_test = test_data

        # Create classifier with one configuration
        clf1 = TabPFNClassifier(n_estimators=2, random_state=42, device="cpu")
        clf1.fit(X_train, y_train)
        predictions1 = clf1.predict_proba(X_test)
        hash1 = get_prediction_hash(predictions1)

        # Create classifier with different configuration
        clf2 = TabPFNClassifier(n_estimators=2, random_state=43, device="cpu")
        clf2.fit(X_train, y_train)
        predictions2 = clf2.predict_proba(X_test)
        hash2 = get_prediction_hash(predictions2)

        # Verify hash detects the configuration change
        assert (
            hash1 != hash2
        ), "Hash function failed to detect change in classifier configuration"

    def test_regressor_hash_detects_data_changes(self, test_data):
        """Verify hash function detects changes in regression input data."""
        X_train, X_test, y_train, y_test = test_data

        # Create regressor with fixed settings
        reg = TabPFNRegressor(n_estimators=2, random_state=42, device="cpu")
        reg.fit(X_train, y_train)

        # Get predictions on original data
        original_predictions = reg.predict(X_test)
        original_hash = get_prediction_hash(original_predictions)

        # Modify test data slightly
        X_test_modified = X_test.copy()
        X_test_modified[0, 0] += 0.1  # Small change to first feature

        # Get predictions on modified data
        modified_predictions = reg.predict(X_test_modified)
        modified_hash = get_prediction_hash(modified_predictions)

        # Verify hash detects the change
        assert (
            original_hash != modified_hash
        ), "Hash function failed to detect change in regression data"

    def test_regressor_hash_detects_model_changes(self, test_data):
        """Verify hash function detects changes in regressor configuration."""
        X_train, X_test, y_train, y_test = test_data

        # Create regressor with one configuration
        reg1 = TabPFNRegressor(n_estimators=2, random_state=42, device="cpu")
        reg1.fit(X_train, y_train)
        predictions1 = reg1.predict(X_test)
        hash1 = get_prediction_hash(predictions1)

        # Create regressor with different configuration
        reg2 = TabPFNRegressor(n_estimators=2, random_state=43, device="cpu")
        reg2.fit(X_train, y_train)
        predictions2 = reg2.predict(X_test)
        hash2 = get_prediction_hash(predictions2)

        # Verify hash detects the configuration change
        assert (
            hash1 != hash2
        ), "Hash function failed to detect change in regressor configuration"


# Helper function to generate reference hashes from current model outputs
def update_reference_hashes():
    """Generate and print reference hashes for current model predictions.

    Run this function manually when intentionally updating reference hashes:
    ```
    python -c "from tests.test_consistency import update_reference_hashes; \
update_reference_hashes()"
    ```

    Steps to update reference hashes:
    1. Make your changes to the TabPFN code
    2. Verify these changes improve model performance on benchmarks
    3. Run this function to generate new reference hashes
    4. Update the REFERENCE_HASHES dictionary in test_consistency.py
    5. Document the improvements in your PR description
    """
    reference_hashes = {}

    # Iris dataset
    random_state = check_random_state(42)
    X, y = sklearn.datasets.load_iris(return_X_y=True, as_frame=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(len(X))
    random_state.shuffle(indices)
    X, y = X[indices], y[indices]
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    clf = TabPFNClassifier(n_estimators=2, random_state=42, device="cpu")
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)
    reference_hashes["iris_classifier"] = get_prediction_hash(predictions)

    # Breast Cancer dataset
    random_state = check_random_state(42)
    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True, as_frame=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(len(X))
    random_state.shuffle(indices)
    X, y = X[indices], y[indices]
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    clf = TabPFNClassifier(n_estimators=2, random_state=42, device="cpu")
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)
    reference_hashes["breast_cancer_classifier"] = get_prediction_hash(predictions)

    # Boston Housing dataset
    try:
        from sklearn.datasets import fetch_openml

        housing = fetch_openml(data_id=531, as_frame=False)
        X, y = housing.data, housing.target
    except (ImportError, ValueError):
        from sklearn.datasets import fetch_california_housing

        housing = fetch_california_housing(as_frame=False)
        X, y = housing.data, housing.target

    random_state = check_random_state(42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(len(X))
    random_state.shuffle(indices)
    X, y = X[indices], y[indices]
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    reg = TabPFNRegressor(n_estimators=2, random_state=42, device="cpu")
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    reference_hashes["boston_regressor"] = get_prediction_hash(predictions)

    # Diabetes dataset
    random_state = check_random_state(42)
    X, y = sklearn.datasets.load_diabetes(return_X_y=True, as_frame=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(len(X))
    random_state.shuffle(indices)
    X, y = X[indices], y[indices]
    train_size = int(0.7 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    reg = TabPFNRegressor(n_estimators=2, random_state=42, device="cpu")
    reg.fit(X_train, y_train)
    predictions = reg.predict(X_test)
    reference_hashes["diabetes_regressor"] = get_prediction_hash(predictions)

    # Print reference hashes in format that can be directly copied into the code
    for _key, _value in reference_hashes.items():
        pass

    return reference_hashes


if __name__ == "__main__":
    # This makes it easier to run the hash update function
    update_reference_hashes()
