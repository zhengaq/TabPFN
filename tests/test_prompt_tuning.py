# tests/test_prompt_tuning.py
from __future__ import annotations

import pytest

# TODO: Implement tests for the prompt-tuning workflow.
# These tests should cover:
# 1.  Initialization of TabPFNClassifier with differentiable_input=True.
# 2.  Using predict_proba_tensor and ensuring gradients can flow back
#     to the prompt tensors (requires_grad=True).
# 3.  A basic prompt-tuning loop similar to examples/tabpfn_prompttune.py,
#     verifying that prompt tensors are updated and loss decreases.
# 4.  Testing interaction with different devices (CPU/GPU).
# 5.  Testing with DifferentiableZNormStep if it's specifically used in this path.
# 6.  Edge cases (e.g., different numbers of prompt samples).


@pytest.mark.skip(reason="Prompt tuning tests not yet implemented")
def test_prompt_tuning_basic_loop():
    """Test a basic prompt-tuning loop with gradient updates."""
    # Setup synthetic data
    # Initialize classifier with differentiable_input=True
    # Create prompt tensors with requires_grad=True
    # Setup optimizer
    # Run a few optimization steps:

    # Assert that prompt tensors have changed
    # Assert that loss has decreased (or gradients exist)


@pytest.mark.skip(reason="Prompt tuning tests not yet implemented")
def test_predict_proba_tensor_gradients():
    """Verify that gradients flow back to input tensors when using
    predict_proba_tensor with differentiable_input=True.
    """
    # Setup synthetic data
    # Initialize classifier with differentiable_input=True
    # Create input tensor X_test with requires_grad=True
    # Fit the model with some prompt tensors (or standard fit)
