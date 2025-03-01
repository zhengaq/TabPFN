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
   - Computes statistical summaries of model predictions
   - Compares statistics against known reference values

2. `TestStatsRobustness`: Verifies that our statistical approach correctly detects changes
   - Confirms that statistics change when input data changes significantly
   - Confirms that statistics change when model configuration changes
   - Tests both classifiers and regressors

### Cross-Platform Consistency

TabPFN uses a statistical approach to verify model consistency that works reliably across different:
- Operating systems (Windows, macOS, Linux)
- Hardware architectures (x86, ARM, etc.)
- Python versions
- NumPy versions

Instead of relying on exact hash values (which can be brittle due to platform-specific floating-point differences),
our approach computes key statistical properties of model predictions:
1. **Distribution statistics** - min, max, mean, median, standard deviation
2. **Percentiles** - p10, p25, p75, p90 capture the distribution shape
3. **Output shape** - ensures the dimensionality remains consistent

The tests allow for small variations (1% relative tolerance) in these statistics to account for 
inevitable floating-point differences across platforms, while still detecting any meaningful
changes in model behavior. This approach provides a good balance between:
- Being strict enough to catch real regressions or changes in model behavior
- Being flexible enough to avoid false positives from harmless platform-specific variations

### When a Consistency Test Fails

When a consistency test fails, it means the model's behavior has changed. The failure message includes:

1. The expected statistic that differs from the reference
2. The actual statistic from your current code
3. The difference between them
4. Instructions for how to proceed

### Managing Intentional Model Changes

If you're making intentional changes to the model that should improve performance:

1. **Verify the improvement:**
   - Benchmark the new model against the old one on standard datasets
   - Document the performance improvements with metrics
   - Consider running on the established AutoML benchmark suite

2. **Update the reference statistics:**
   ```python
   python -c "from tests.test_consistency import update_reference_stats; update_reference_stats()"
   ```

3. **Document the changes in your PR:**
   - Explain what changes were made to the model
   - Provide benchmark results showing the improvement
   - Explain why this change is beneficial

4. **Update the REFERENCE_STATS dictionary:**
   - Replace the values in `test_consistency.py` with the newly generated statistics
   - Include this change in your PR

### Rules for Model Changes

Model changes should:

1. Be intentional and well-understood (not accidental side effects)
2. Improve overall performance on standard benchmarks
3. Not break backward compatibility for users without good reason
4. Be thoroughly documented with evidence of improvement

By following these guidelines, we ensure that TabPFN evolves in a controlled and beneficial way, providing users with a reliable and continually improving foundation model.