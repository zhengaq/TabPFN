# TabPFN Tests

This directory contains test files for the TabPFN project.

## Test Files

- `test_classifier_interface.py`: Tests for the TabPFNClassifier interface
- `test_regressor_interface.py`: Tests for the TabPFNRegressor interface 
- `test_text_na_handling.py`: Tests for text and NA value handling
- `test_utils.py`: Tests for utility functions
- `test_consistency.py`: Tests to ensure prediction consistency across code changes

## Model Consistency Testing

The `test_consistency.py` file contains tests that verify TabPFN models produce consistent predictions across code changes. These tests help ensure that:

1. Changes to the codebase don't unexpectedly alter model behavior
2. Core algorithms remain stable and reproducible 
3. Any intentional changes to model behavior are explicitly acknowledged

### How It Works

The consistency tests use small, fixed datasets with reproducible random seeds to ensure consistency across code changes. Each test:

1. Creates a TabPFN model with fixed settings
2. Fits it to a reproducible dataset
3. Gets predictions using a standardized process
4. Compares those predictions to previously saved reference values

If predictions change beyond tolerance thresholds, the test fails, indicating that code changes have altered model behavior.

### Platform Compatibility

TabPFN models can produce slightly different predictions across platforms due to:
- Different CPU architectures (x86 vs ARM)
- Different operating systems (Linux, macOS, Windows)
- Different Python versions
- Different underlying numerical libraries

For this reason:
1. Reference predictions are platform-specific (stored in `reference_predictions/`)
2. Platform information is tracked in `reference_predictions/platform_metadata.json`
3. Tests only run on matching platforms by default

### CI Compatibility

For continuous integration (CI), we test against specific platform configurations:
- Linux, Windows, and macOS (Darwin)
- Python 3.9 and 3.12

To ensure reliable CI testing:
1. Reference values should be generated on one of the CI platforms
2. CI tests will fail with instructions if reference platform doesn't match

### Checking Your Platform

To check if your platform matches a CI configuration and/or the reference platform:

```bash
python tests/test_consistency.py --print-platform
```

This command will show:
- Your current platform information
- Whether it matches a CI configuration
- Reference platform information
- Whether the reference platform matches a CI configuration
- Whether your platform matches the reference platform

### Updating Reference Values

Reference values should only be updated when necessary (e.g., legitimate algorithm improvements). Ideally, generate them on a CI-compatible platform:

```bash
python tests/test_consistency.py --update-reference
```

This command will:
- Check if your platform matches a CI configuration
- Warn and ask for confirmation if it doesn't
- Generate new reference predictions
- Update the platform metadata

### Running Tests on Different Platforms

If you're not on the reference platform but need to run the tests:

```bash
FORCE_CONSISTENCY_TESTS=1 pytest tests/test_consistency.py
```

In CI environments, tests will fail with helpful instructions if the reference platform doesn't match a CI configuration.

### When a Consistency Test Fails

When a test fails, it means the model's predictions have changed. The failure message includes:

1. The test case that failed
2. The tolerance settings that were exceeded
3. Instructions to either:
   - Fix the code to restore the original behavior
   - Update reference values if the change is intentional

### Guidelines for Model Changes

Model changes should:
1. Be intentional and well-understood (not accidental side effects)
2. Improve performance on standard benchmarks
3. Maintain backward compatibility unless there's good reason not to
4. Be clearly documented with evidence of improvement

By following these guidelines, we ensure that TabPFN evolves in a controlled and beneficial way.