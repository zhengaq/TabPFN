# TabPFN Tests

This directory contains test files for the TabPFN project.

## Test Files

- `test_classifier_interface.py`: Tests for the TabPFNClassifier interface
- `test_regressor_interface.py`: Tests for the TabPFNRegressor interface
- `test_utils.py`: Tests for utility functions
- `test_consistency.py`: Tests to ensure prediction consistency across code changes

## Model Consistency Testing

The `test_consistency.py` file contains tests that verify TabPFN models produce consistent predictions across code changes. These tests help ensure that:

1. Changes to the codebase don't unexpectedly alter model behavior
2. Core algorithms remain stable and reproducible 
3. Any intentional changes to model behavior are explicitly acknowledged and verified

### How It Works

The tests are organized into two classes:

1. `TestModelConsistency`: Verifies that TabPFN models produce consistent predictions
   - Uses standardized datasets (Iris, Breast Cancer, Boston Housing, Diabetes)
   - Creates models with fixed random seeds
   - Computes hashes of model predictions
   - Compares hashes against known reference values

2. `TestHashRobustness`: Verifies that our hash function correctly detects changes
   - Confirms that the hash changes when input data changes
   - Confirms that the hash changes when model configuration changes
   - Tests both classifiers and regressors

### When a Consistency Test Fails

When a consistency test fails, it means the model's behavior has changed. The failure message includes:

1. The expected hash from the reference
2. The actual hash from your current code
3. Instructions for how to proceed

### Managing Intentional Model Changes

If you're making intentional changes to the model that should improve performance:

1. **Verify the improvement:**
   - Benchmark the new model against the old one on standard datasets
   - Document the performance improvements with metrics
   - Consider running on the established AutoML benchmark suite

2. **Update the reference hashes:**
   ```python
   python -c "from tests.test_consistency import update_reference_hashes; update_reference_hashes()"
   ```

3. **Document the changes in your PR:**
   - Explain what changes were made to the model
   - Provide benchmark results showing the improvement
   - Explain why this change is beneficial

4. **Update the REFERENCE_HASHES dictionary:**
   - Replace the values in `test_consistency.py` with the newly generated hashes
   - Include this change in your PR

### Rules for Model Changes

Model changes should:

1. Be intentional and well-understood (not accidental side effects)
2. Improve overall performance on standard benchmarks
3. Not break backward compatibility for users without good reason
4. Be thoroughly documented with evidence of improvement

By following these guidelines, we ensure that TabPFN evolves in a controlled and beneficial way, providing users with a reliable and continually improving foundation model.