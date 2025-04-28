
# Klemens Testing TODOs (25.04.2025)



~~- * The logic within DatasetCollectionWithPreprocessing.__getitem__ is quite complex, especially handling the conditional logic for regression. Add comments to clarify the flow and the structure of the returned tuple.~~
    -> added more documentations


- ~~Testing on GPU now~~ -> everything runs

~~- initialize_tabpfn_model: Ensure that when pre-loaded ModelSpecs are passed, the model is correctly moved to the target device if specified later in the __init__ or fit methods. The current logic returns the model as-is from the spec.~~
    -> moves model_ and (renreomalized corterion to device, in _initialize_model_variables)
    -> Moved the devices + passes GPU tests


- use_torch_inference_mode: Good addition for flexibility (allowing backprop). Ensure docstrings clearly state which engines support this (BatchedNoPreprocessing and CachePreprocessing seem to, CacheKV likely doesn't due to caching?). The toggling between model.cpu() and model.to(device) needs care, especially in loops, to avoid unnecessary data transfers or potential memory issues.
    ~~-> use_torch_inference_mode~~
    ~~-> Double check the torch_inference_mode~~
    ~~-> more documentation in inference.py~~
    TODO: -> Check toggling model.cpu() and model.to(device) 


~~- comments for: hyperparams["optimization_space"]~~

~~- sider abstracting common setup/evaluation logic used across finetuning examples into shared utility functions~~
    -> define a new function that handles the model copying

~~- eval_test / eval_test_regression_standard: The use of copy.deepcopy(clf.model_) is crucial and correct to prevent the evaluation step from modifying the model being finetuned. Explicitly comment why this is necessary.~~
    -> Added more documenation
    -> Moved the model copying into a new file


        
* Update relevant docstrings 
    -> ~~TabPFNClassifier.fit ~~
    -> TabPFNRegressor.fit
    -> preprocessing classes 
        - ~~DatasetCollectionWithPreprocessing~~
    -> ~~Inferene Engines (got new use_torch_inference_mode) ~~
        - ~~InferenceEngineBatchedNoPreprocessing~~ updates documentations
        - ~~InferenceEngineCachePreprocessing~~
        
* Explain the differentiable_input and ignore_pretraining_limits parameters more thoroughly.:
    -> differentiable_input: Go through this
    -> ignore_pretraining_limits
    

- Documentation: Significant new features like finetuning, prompt tuning, batched inference, and differentiable preprocessing require documentation updates.
    - Fix Prompt Tuning
    - Test Differentiable input:
        - Classifier
        - _initialize_model_variables
        - update_encoder_params

 ## Testing: 
 
 This is the most critical area needing attention. Given the significant new features (finetuning loops, differentiable paths, batched inference, new preprocessing steps, tensor inputs), comprehensive unit and integration tests are essential.

 * ~~Make test case for CUDA and cpu, iterate over different settings~~

* ~~ Fox failing tests and linting ~~
* Add tests for 
    - ~~ fit_from_preprocessed ~~
    - ~~fit_from_preprocessed (GPU)~~ 
    - ~~predict_proba_from_preprocessed~~
    - predict_proba_tensor and and compare to predict_proba (Prompt Tuning stuff)
* Add tests for the 
    - InferenceEngineBatchedNoPreprocessing 
    ~~ - collate_for_tabpfn_dataset function. ~~
* Add tests for the differentiable preprocessing path 
    (DifferentiableZNormStep, differentiable=True config).
    -> ~~ DifferentiableZNormStep~~
    -> differentiable=True config   
* Add tests for utils.py functions: 
    - ~~split_large_data~~
    - ~~pad_tensors~~ 
    - update_encoder_params.



- ~~Make config list of tuples for datasets into a new class~~
- ~~Write tests for DataSet Class ~~
- ~~Test Collator~~
- ~~Test Classifier.get_preprocessed_datasets~~ (bit more though)
- Test Classifier.fit_from_preprocessed
- Test classifier.predict_proba_from_preprocessed


- test that eval_test does not modify the underlying class

- Test Regressor



* Ensure existing tests pass and potentially add tests covering interactions between new and old components.
    -> Full FT setup




Current tests: 

### File: `test_finetuning_classifier.py`

* **`test_tabpfn_finetune_basic_runs()`:** Checks if the basic `TabPFNClassifier` fitting and prediction process runs without errors and produces outputs of the correct shape.
* **`test_eval_test_function()`:** Verifies that the `eval_test` helper function from the examples correctly computes accuracy and log-likelihood for a trained classifier.
* **`test_datasetcollectionwithpreprocessing_classification_single_dataset()`:** Tests the `get_preprocessed_datasets` method for a single dataset, ensuring the output `DatasetCollectionWithPreprocessing` has the correct structure and data shapes.
* **`test_datasetcollectionwithpreprocessing_classification_multiple_datasets()`:** Validates the `get_preprocessed_datasets` method when processing multiple input datasets simultaneously, checking the structure and shapes for each dataset's output.
* **`test_get_preprocessed_datasets_basic()`:** Performs a basic check that `get_preprocessed_datasets` returns a valid collection object with the expected basic attributes and item structure.
* **`test_dataset_and_collator_with_dataloader_uniform()`:** Ensures that the `DatasetCollectionWithPreprocessing` works correctly with a `DataLoader` and the custom collate function for datasets of uniform size, validating the batch format.
* **`test_dataset_and_collator_with_dataloader_variable()`:** Verifies the `DatasetCollectionWithPreprocessing`, `DataLoader`, and collate function integration for datasets with varying sizes and class counts.
* **`test_tabpfn_finetune_from_preprocessed_runs()`:** Tests the fine-tuning workflow starting from preprocessed data (`fit_from_preprocessed`, `predict_proba_from_preprocessed`), including loss calculation and gradient backpropagation.

### File: `test_ft_utils.py`

* **`test_pad_tensors_2d_and_1d()`:** Checks the `pad_tensors` utility for correctly padding lists of both 1D and 2D tensors to uniform shapes using a specified padding value.
* **`test_split_large_data()`:** Validates the `split_large_data` utility function for correctly chunking large datasets based on a maximum chunk size, including handling edge cases.