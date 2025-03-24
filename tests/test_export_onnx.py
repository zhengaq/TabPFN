from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.misc.compile_to_onnx import compile_onnx_models


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_onnx_missing_model_error():
    """Test that appropriate error is raised when trying to
    use ONNX with a missing model. Here we specify a model path
    that does not exist to simulate the case where the model
    has not been compiled.
    """
    if os.name == "nt":
        pytest.skip("ONNX export is not tested on Windows")
    if sys.version_info >= (3, 13):
        pytest.xfail("ONNX is not yet supported on Python 3.13")

    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("ONNX or ONNX Runtime not available")

    # Generate synthetic data
    rng = np.random.default_rng()
    X = rng.standard_normal((50, 10)).astype(np.float32)
    y = rng.integers(0, 2, size=50)

    # Try to use ONNX backend when model doesn't exist
    classifier = TabPFNClassifier(
        device="cpu", use_onnx=True, model_path="/fake_dir/tabpfn_classifier_v2.ckpt"
    )

    # Expect a FileNotFoundError with a specific message
    with pytest.raises(
        FileNotFoundError,
        match=(
            r"ONNX model not found at:.*please compile the model by "
            r"running.*compile_onnx_models\(\)"
        ),
    ):
        classifier.fit(X, y)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
def test_onnx_export_and_inference():
    """Test that TabPFN models can be exported to ONNX
    and produce correct predictions.
    """
    if os.name == "nt":
        pytest.skip("ONNX export is not tested on Windows")
    if sys.version_info >= (3, 13):
        pytest.xfail("ONNX is not yet supported on Python 3.13")

    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("ONNX or ONNX Runtime not available")

    # Compile the model to ONNX format (using default output directory)
    compile_onnx_models(skip_test=True)

    # Generate synthetic data for testing
    n_samples = 100
    n_features = 10
    rng = np.random.default_rng()
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_samples)

    # Split into train/test
    train_size = 80
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, _y_test = y[:train_size], y[train_size:]

    # Test with PyTorch backend
    classifier_torch = TabPFNClassifier(device="cpu", use_onnx=False)
    classifier_torch.fit(X_train, y_train)

    # Get predictions with PyTorch backend
    torch_probs = classifier_torch.predict_proba(X_test)
    torch_preds = classifier_torch.predict(X_test)

    # Test with ONNX backend
    classifier_onnx = TabPFNClassifier(device="cpu", use_onnx=True)
    classifier_onnx.fit(X_train, y_train)

    # Get predictions with ONNX backend
    onnx_probs = classifier_onnx.predict_proba(X_test)
    onnx_preds = classifier_onnx.predict(X_test)

    # Check that the predictions roughly match
    np.testing.assert_allclose(torch_probs, onnx_probs, rtol=1e-2, atol=1e-2)
    np.testing.assert_array_equal(torch_preds, onnx_preds)

    # same for regressor
    regressor_torch = TabPFNRegressor(device="cpu", use_onnx=False)
    regressor_torch.fit(X_train, y_train)
    regressor_onnx = TabPFNRegressor(device="cpu", use_onnx=True)
    regressor_onnx.fit(X_train, y_train)

    torch_preds = regressor_torch.predict(X_test)
    onnx_preds = regressor_onnx.predict(X_test)

    # Check that the predictions roughly match
    np.testing.assert_allclose(torch_preds, onnx_preds, rtol=1e-2, atol=1e-2)
