
# Klemens Testing TODOs (25.04.2025)



- * The logic within DatasetCollectionWithPreprocessing.__getitem__ is quite complex, especially handling the conditional logic for regression. Add comments to clarify the flow and the structure of the returned tuple.

- Test Regressor

- initialize_tabpfn_model: Ensure that when pre-loaded ModelSpecs are passed, the model is correctly moved to the target device if specified later in the __init__ or fit methods. The current logic returns the model as-is from the spec.


- use_torch_inference_mode: Good addition for flexibility (allowing backprop). Ensure docstrings clearly state which engines support this (BatchedNoPreprocessing and CachePreprocessing seem to, CacheKV likely doesn't due to caching?). The toggling between model.cpu() and model.to(device) needs care, especially in loops, to avoid unnecessary data transfers or potential memory issues.

~~- comments for: hyperparams["optimization_space"]~~

- sider abstracting common setup/evaluation logic used across finetuning examples into shared utility functions

- eval_test / eval_test_regression_standard: The use of copy.deepcopy(clf.model_) is crucial and correct to prevent the evaluation step from modifying the model being finetuned. Explicitly comment why this is necessary.


- Documentation: Significant new features like finetuning, prompt tuning, batched inference, and differentiable preprocessing require documentation updates.
* Update relevant docstrings (e.g., TabPFNClassifier.fit, TabPFNRegressor.fit, preprocessing classes, inference engines) with changes.
* Explain the differentiable_input and ignore_pretraining_limits parameters more thoroughly.


 ## Testing: 
 
 This is the most critical area needing attention. Given the significant new features (finetuning loops, differentiable paths, batched inference, new preprocessing steps, tensor inputs), comprehensive unit and integration tests are essential.

* ~~ Fox failing tests and linting ~~
* Add tests for 
    - fit_from_preprocessed 
    - predict_proba_from_preprocessed.
* Add tests for the 
    - InferenceEngineBatchedNoPreprocessing 
    ~~ - collate_for_tabpfn_dataset function. ~~
* Add tests for the differentiable preprocessing path 
    (DifferentiableZNormStep, differentiable=True config).
* Add tests for utils.py functions: 
- split_large_data,
- ~~pad_tensors,~~ 
- update_encoder_params.
* Ensure existing tests pass and potentially add tests covering interactions between new and old components.


- ~~Make config list of tuples for datasets into a new class~~
- ~~Write tests for DataSet Class ~~
- ~~Test Collator~~
- ~~Test Classifier.get_preprocessed_datasets~~ (bit more though)
- Test Classifier.fit_from_preprocessed
- Test classifier.predict_proba_from_preprocessed