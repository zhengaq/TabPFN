"""TabPFNRegressor class.

!!! example
    ```python
    import sklearn.datasets
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor()
    X, y = sklearn.datasets.make_regression(n_samples=50, n_features=10)

    model.fit(X, y)
    predictions = model.predict(X)
    ```
"""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import typing
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import Self, TypedDict, overload

import numpy as np
import torch
from sklearn import config_context
from sklearn.base import (
    BaseEstimator,
    RegressorMixin,
    TransformerMixin,
    check_is_fitted,
)

from tabpfn.base import (
    check_cpu_warning,
    create_inference_engine,
    determine_precision,
    initialize_tabpfn_model,
)
from tabpfn.config import ModelInterfaceConfig
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.model.preprocessing import (
    ReshapeFeatureDistributionsStep,
)
from tabpfn.preprocessing import ( 
    DatasetCollectionWithPreprocessing_Regressor, # To be used by get_preprocessed_datasets
    EnsembleConfig,
    PreprocessorConfig,
    RegressorEnsembleConfig,
    default_regressor_preprocessor_configs,
)
from tabpfn.utils import (
    _fix_dtypes,
    _get_embeddings,
    _get_ordinal_encoder,
    _process_text_na_dataframe,
    _transform_borders_one,
    infer_categorical_features,
    infer_device_and_type,
    infer_random_state,
    split_large_data, # Used in get_preprocessed_datasets
    translate_probs_across_borders,
    update_encoder_outlier_params, # Renamed from update_encoder_outlier_params
    validate_X_predict,
    validate_Xy_fit,
)

# NEW: Import the specialized inference engine
from tabpfn.inference import InferenceEngineBatchedNoPreprocessing

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from torch.types import _dtype

    from tabpfn.constants import (
        XType,
        YType,
    )
    from tabpfn.inference import (
        InferenceEngine,
    )
    from tabpfn.model.config import InferenceConfig

    try:
        from sklearn.base import Tags
    except ImportError:
        Tags = Any


# TypedDict definitions for prediction outputs
class MainOutputDict(TypedDict):
    """Dictionary containing the main output types from the TabPFN regressor."""

    mean: np.ndarray
    median: np.ndarray
    mode: np.ndarray
    quantiles: list[np.ndarray]


class FullOutputDict(MainOutputDict):
    """Dictionary containing all outputs from the TabPFN regressor."""

    criterion: FullSupportBarDistribution
    logits: torch.Tensor


class TabPFNRegressor(RegressorMixin, BaseEstimator):  
    """TabPFNRegressor class."""

    config_: InferenceConfig
    """The configuration of the loaded model to be used for inference."""

    interface_config_: ModelInterfaceConfig
    """Additional configuration of the interface for expert users."""

    device_: torch.device
    """The device determined to be used."""

    feature_names_in_: npt.NDArray[Any]
    """The feature names of the input data.

    May not be set if the input data does not have feature names,
    such as with a numpy array.
    """

    n_features_in_: int
    """The number of features in the input data used during `fit()`."""

    inferred_categorical_indices_: list[int]
    """The indices of the columns that were inferred to be categorical,
    as a product of any features deemed categorical by the user and what would
    work best for the model.
    """

    n_outputs_: Literal[1]  # We only support single output
    """The number of outputs the model supports. Only 1 for now"""

    bardist_: FullSupportBarDistribution
    """The bar distribution of the target variable, used by the model."""

    renormalized_criterion_: FullSupportBarDistribution
    """The normalized bar distribution used for computing the predictions."""

    use_autocast_: bool
    """Whether torch's autocast should be used."""

    forced_inference_dtype_: _dtype | None
    """The forced inference dtype for the model based on `inference_precision`."""

    executor_: InferenceEngine
    """The inference engine used to make predictions."""

    y_train_mean_: float
    """The mean of the target variable during training."""

    y_train_std: float
    """The standard deviation of the target variable during training."""

    preprocessor_: ColumnTransformer
    """The column transformer used to preprocess the input data to be numeric."""

    # TODO: consider moving the following to constants.py
    _OUTPUT_TYPES_BASIC = ("mean", "median", "mode")
    """The basic output types supported by the model."""
    _OUTPUT_TYPES_QUANTILES = ("quantiles",)
    """The quantiles output type supported by the model."""
    _OUTPUT_TYPES = _OUTPUT_TYPES_BASIC + _OUTPUT_TYPES_QUANTILES
    """The output types supported by the model for the "main" output type."""
    _OUTPUT_TYPES_COMPOSITE = ("full", "main")
    """The composite output types supported by the model."""
    _USABLE_OUTPUT_TYPES = _OUTPUT_TYPES + _OUTPUT_TYPES_COMPOSITE
    """The output types supported by the model."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        n_estimators: int = 8,
        categorical_features_indices: Sequence[int] | None = None,
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = False,
        model_path: str | Path | Literal["auto"] = "auto",
        device: str | torch.device | Literal["auto"] = "auto",
        ignore_pretraining_limits: bool = False,
        inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
        fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
            # NEW: Add 'batched' mode possibility, although user typically won't set this directly.
            # 'fit_from_preprocessed' will use it internally.
            "batched",
        ] = "fit_preprocessors",
        memory_saving_mode: bool | Literal["auto"] | float | int = "auto",
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
        n_jobs: int = -1,
        inference_config: dict | ModelInterfaceConfig | None = None,
        # NEW: Add differentiable_input for consistency, though less used in regression FT
        differentiable_input: bool = False,
    ) -> None:
        """A TabPFN interface for regression.

        Args:
            n_estimators:
                The number of estimators in the TabPFN ensemble. We aggregate the
                predictions of `n_estimators`-many forward passes of TabPFN.
                Each forward pass has (slightly) different input data. Think of this
                as an ensemble of `n_estimators`-many "prompts" of the input data.

            categorical_features_indices:
                The indices of the columns that are suggested to be treated as
                categorical. If `None`, the model will infer the categorical columns.
                If provided, we might ignore some of the suggestion to better fit the
                data seen during pre-training.

                !!! note
                    The indices are 0-based and should represent the data passed to
                    `.fit()`. If the data changes between the initializations of the
                    model and the `.fit()`, consider setting the
                    `.categorical_features_indices` attribute after the model was
                    initialized and before `.fit()`.

            softmax_temperature:
                The temperature for the softmax function. This is used to control the
                confidence of the model's predictions. Lower values make the model's
                predictions more confident. This is only applied when predicting during
                a post-processing step. Set `softmax_temperature=1.0` for no effect.

            average_before_softmax:
                Only used if `n_estimators > 1`. Whether to average the predictions of
                the estimators before applying the softmax function. This can help to
                improve predictive performance when there are many classes or when
                calibrating the model's confidence. This is only applied when
                predicting during a post-processing.

                - If `True`, the predictions are averaged before applying the softmax
                  function. Thus, we average the logits of TabPFN and then apply the
                  softmax.
                - If `False`, the softmax function is applied to each set of logits.
                  Then, we average the resulting probabilities of each forward pass.

            model_path:
                The path to the TabPFN model file, i.e., the pre-trained weights.

                - If `"auto"`, the model will be downloaded upon first use. This
                  defaults to your system cache directory, but can be overwritten
                  with the use of an environment variable `TABPFN_MODEL_CACHE_DIR`.
                - If a path or a string of a path, the model will be loaded from
                  the user-specified location if available, otherwise it will be
                  downloaded to this location.

            ignore_pretraining_limits:
                Whether to ignore the pre-training limits of the model. The TabPFN
                models have been pre-trained on a specific range of input data. If the
                input data is outside of this range, the model may not perform well.
                You may ignore our limits to use the model on data outside the
                pre-training range.

                - If `True`, the model will not raise an error if the input data is
                  outside the pre-training range.
                - If `False`, you can use the model outside the pre-training range, but
                  the model could perform worse.

                !!! note

                    The current pre-training limits are:

                    - 10_000 samples/rows
                    - 500 features/columns

            device:
                The device to use for inference with TabPFN. If `"auto"`, the device is
                `"cuda"` if available, otherwise `"cpu"`.

                See PyTorch's documentation on devices for more information about
                supported devices.

            inference_precision:
                The precision to use for inference. This can dramatically affect the
                speed and reproducibility of the inference. Higher precision can lead to
                better reproducibility but at the cost of speed. By default, we optimize
                for speed and use torch's mixed-precision autocast. The options are:

                - If `torch.dtype`, we force precision of the model and data to be
                  the specified torch.dtype during inference. This can is particularly
                  useful for reproducibility. Here, we do not use mixed-precision.
                - If `"autocast"`, enable PyTorch's mixed-precision autocast. Ensure
                  that your device is compatible with mixed-precision.
                - If `"auto"`, we determine whether to use autocast or not depending on
                  the device type.

            fit_mode:
                Determine how the TabPFN model is "fitted". The mode determines how the
                data is preprocessed and cached for inference. This is unique to an
                in-context learning foundation model like TabPFN, as the "fitting" is
                technically the forward pass of the model. The options are:

                - If `"low_memory"`, the data is preprocessed on-demand during inference
                  when calling `.predict()` or `.predict_proba()`. This is the most
                  memory-efficient mode but can be slower for large datasets because
                  the data is (repeatedly) preprocessed on-the-fly.
                  Ideal with low GPU memory and/or a single call to `.fit()` and
                  `.predict()`.
                - If `"fit_preprocessors"`, the data is preprocessed and cached once
                  during the `.fit()` call. During inference, the cached preprocessing
                  (of the training data) is used instead of re-computing it.
                  Ideal with low GPU memory and multiple calls to `.predict()` with
                  the same training data.
                - If `"fit_with_cache"`, the data is preprocessed and cached once during
                  the `.fit()` call like in `fit_preprocessors`. Moreover, the
                  transformer key-value cache is also initialized, allowing for much
                  faster inference on the same data at a large cost of memory.
                  Ideal with very high GPU memory and multiple calls to `.predict()`
                  with the same training data.

            memory_saving_mode:
                Enable GPU/CPU memory saving mode. This can help to prevent
                out-of-memory errors that result from computations that would consume
                more memory than available on the current device. We save memory by
                automatically batching certain model computations within TabPFN to
                reduce the total required memory. The options are:

                - If `bool`, enable/disable memory saving mode.
                - If `"auto"`, we will estimate the amount of memory required for the
                  forward pass and apply memory saving if it is more than the
                  available GPU/CPU memory. This is the recommended setting as it
                  allows for speed-ups and prevents memory errors depending on
                  the input data.
                - If `float` or `int`, we treat this value as the maximum amount of
                  available GPU/CPU memory (in GB). We will estimate the amount
                  of memory required for the forward pass and apply memory saving
                  if it is more than this value. Passing a float or int value for
                  this parameter is the same as setting it to True and explicitly
                  specifying the maximum free available memory

                !!! warning
                    This does not batch the original input data. We still recommend to
                    batch this as necessary if you run into memory errors! For example,
                    if the entire input data does not fit into memory, even the memory
                    save mode will not prevent memory errors.

            random_state:
                Controls the randomness of the model. Pass an int for reproducible
                results and see the scikit-learn glossary for more information.
                If `None`, the randomness is determined by the system when calling
                `.fit()`.

                !!! warning
                    We depart from the usual scikit-learn behavior in that by default
                    we provide a fixed seed of `0`.

                !!! note
                    Even if a seed is passed, we cannot always guarantee reproducibility
                    due to PyTorch's non-deterministic operations and general numerical
                    instability. To get the most reproducible results across hardware,
                    we recommend using a higher precision as well (at the cost of a
                    much higher inference time). Likewise, for scikit-learn, consider
                    passing `USE_SKLEARN_16_DECIMAL_PRECISION=True` as kwarg.

            n_jobs:
                The number of workers for tasks that can be parallelized across CPU
                cores. Currently, this is used for preprocessing the data in parallel
                (if `n_estimators > 1`).

                - If `-1`, all available CPU cores are used.
                - If `int`, the number of CPU cores to use is determined by `n_jobs`.

            inference_config:
                For advanced users, additional advanced arguments that adjust the
                behavior of the model interface.
                See [tabpfn.constants.ModelInterfaceConfig][] for details and options.

                - If `None`, the default ModelInterfaceConfig is used.
                - If `dict`, the key-value pairs are used to update the default
                  `ModelInterfaceConfig`. Raises an error if an unknown key is passed.
                - If `ModelInterfaceConfig`, the object is used as the configuration.
            
            differentiable_input: (NEW)
                If true, preprocessing attempts to be end-to-end differentiable.
                Less relevant for standard regression fine-tuning compared to prompt-tuning.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.model_path = model_path
        self.device = device
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision: torch.dtype | Literal["autocast", "auto"] = (
            inference_precision
        )
        self.fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"] = (
            fit_mode
        )
        #self.fit_mode = fit_mode
        self.memory_saving_mode: bool | Literal["auto"] | float | int = (
            memory_saving_mode
        )
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.inference_config = inference_config
        # NEW: Store differentiable_input
        self.differentiable_input = differentiable_input

    # TODO: We can remove this from scikit-learn lower bound of 1.6
    def _more_tags(self) -> dict[str, Any]:
        return {
            "allow_nan": True,
        }

    def __sklearn_tags__(self) -> Tags:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.estimator_type = "regressor"
        return tags
    
    # --- NEW: Fine-tuning specific methods START ---

    def get_preprocessed_datasets(
        self,
        X: XType | list[XType],
        y: YType | list[YType],
        split_fn,
        max_data_size: None | int = 10000,
    ) -> Dataset:
        """Get a torch.utils.data.Dataset for fine-tuning or evaluation.
        Each item corresponds to a train/test split of a dataset, preprocessed
        according to TabPFN's ensemble configurations.

        Args:
            X: Input features (single dataset or list of datasets).
            y: Input targets (single dataset or list of datasets).
            split_fn: A function (e.g., from functools.partial(train_test_split))
                      to dissect a dataset into train and test partitions.
            max_data_size: Maximum allowed number of samples in one dataset chunk
                           before splitting. If None, datasets are not chunked.

        Returns:
            A PyTorch Dataset object yielding preprocessed data tuples.
        """
        if not isinstance(X, list):
            X = [X]
        if not isinstance(y, list):
            y = [y]
        assert len(X) == len(y), "X and y lists must have the same length."

        # Initialize model variables if not already done
        if not hasattr(self, "model_") or self.model_ is None:
            # Note: This also sets self.y_train_mean_ and self.y_train_std_ based on the *first*
            # dataset encountered here if fit() hasn't been called. Be mindful if using multiple
            # datasets with different scales without calling fit first. Calling fit() first is recommended.
            byte_size, rng = self._initialize_model_variables(X[0], y[0]) # Initialize based on first dataset if needed
        else:
            # If fit was called, these are already set
            _, rng = infer_random_state(self.random_state)
            # Need byte_size if fine-tuning starts without calling .fit()
            if not hasattr(self, 'forced_inference_dtype_'):
                 _, _, byte_size = determine_precision(self.inference_precision, self.device_)


        # Chunk large datasets
        X_split, y_split = [], []
        for X_item, y_item in zip(X, y):
            if max_data_size is not None and len(X_item) > max_data_size:
                Xparts, yparts = split_large_data(X_item, y_item, max_data_size)
                X_split.extend(Xparts)
                y_split.extend(yparts)
            else:
                X_split.append(X_item)
                y_split.append(y_item)
        X_all, y_all = X_split, y_split

        # Prepare configurations for each dataset/chunk
        config_collection = []
        for X_item, y_item in zip(X_all, y_all):
            # This initializes preprocessing based on THIS specific X_item, y_item
            # Crucially, it generates the ensemble configs and infers categoricals
            # Note: y standardization happens *inside* the Dataset __getitem__ using self.y_train_mean/std_
            configs, X_mod, y_mod_standardized, y_mean, y_std = self._initialize_dataset_preprocessing(
                X_item, y_item, rng
            )
            # Store the original X and y along with configs and cat_indices
            # The actual split, preprocessing, and standardization happens in DatasetCollectionWithPreprocessing_Regressor
            config_collection.append(
                [configs, X_mod, y_item, self.inferred_categorical_indices_, y_mean, y_std] # Pass original y and stats
            )

        # Pass mean/std to the dataset so it can standardize y_train within __getitem__
        # and potentially pass them to the collate function if needed elsewhere (though accessing via self is easier)
        return DatasetCollectionWithPreprocessing_Regressor(split_fn, rng, config_collection,
                                                  y_stats={'mean': self.y_train_mean_, 'std': self.y_train_std_})

    def _initialize_model_variables(self, X_init: XType, y_init: YType) -> tuple[int, np.random.Generator]:
        """Initialize model, device, precision, configs, and y-stats."""
        static_seed, rng = infer_random_state(self.random_state)

        # Load the model and config
        # Crucially load the 'regressor' model and the bardist
        self.model_, self.config_, self.bardist_ = initialize_tabpfn_model(
            model_path=self.model_path,
            which="regressor",
            fit_mode=self.fit_mode, # Use the instance's fit_mode
            static_seed=static_seed,
        )

        # Determine device and precision
        self.device_ = infer_device_and_type(self.device)
        (self.use_autocast_, self.forced_inference_dtype_, byte_size) = (
            determine_precision(self.inference_precision, self.device_)
        )

        # Build the interface_config
        self.interface_config_ = ModelInterfaceConfig.from_user_input(
            inference_config=self.inference_config,
        )

        # Update encoder params (e.g., outlier handling)
        outlier_removal_std = self.interface_config_.OUTLIER_REMOVAL_STD
        if outlier_removal_std == "auto":
            outlier_removal_std = (
                self.interface_config_._REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD
            )
        update_encoder_outlier_params( # Use the renamed function if available, or original one
            model=self.model_,
            remove_outliers_std=outlier_removal_std,
            seed=static_seed,
            inplace=True,
            # Pass differentiable_input if the function supports it
            # differentiable_input=self.differentiable_input
        )

        

        # Calculate and store y mean/std based on initial data provided
        # This ensures these values are set even if .fit() isn't called before get_preprocessed_datasets
        # Use the *raw* y_init for calculating stats
        temp_y = validate_Xy_fit(X_init, y_init, 
                                self, ensure_y_numeric=True,
                                max_num_features=self.interface_config_.MAX_NUMBER_OF_FEATURES,
                                max_num_samples=self.interface_config_.MAX_NUMBER_OF_SAMPLES,
                )[1] # Basic validation
        self.y_train_mean_ = np.mean(temp_y).item()
        self.y_train_std_ = np.std(temp_y).item() + 1e-20 # Add epsilon
        self.renormalized_criterion_ = FullSupportBarDistribution(
            self.bardist_.borders * self.y_train_std_ + self.y_train_mean_,
        ).float()


        return byte_size, rng
    


    def _initialize_dataset_preprocessing(
        self, X: XType, y: YType, rng
    ) -> tuple[list[RegressorEnsembleConfig], XType, YType, float, float]:
        """Prepare ensemble configs and validate X, y for one dataset/chunk."""

        # Validate input data (similar to fit, but no need to set all attributes yet)
        X_proc, y_proc, feature_names_in, n_features_in = validate_Xy_fit(
            X,
            y,
            estimator=self,
            ensure_y_numeric=True, # Ensure y is numeric for regression
            max_num_samples=self.interface_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.interface_config_.MAX_NUMBER_OF_FEATURES,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )
        # We'll use X_proc for categorical inference, but return original X,y
        # The actual preprocessing happens inside the Dataset __getitem__

        check_cpu_warning(self.device_, X_proc) # Use validated X

        # Handle categorical features (same as in fit)
        X_cat_infer = _fix_dtypes(X_proc, cat_indices=self.categorical_features_indices)
        # We only fit the ordinal encoder here if needed, it's applied later
        ord_encoder = _get_ordinal_encoder()
        X_cat_infer = _process_text_na_dataframe(X_cat_infer, ord_encoder=ord_encoder, fit_encoder=True)

        # Infer categorical features based on the processed data
        # Store on self, as it's assumed consistent across datasets in a fine-tuning run
        self.inferred_categorical_indices_ = infer_categorical_features(
            X=X_cat_infer,
            provided=self.categorical_features_indices,
            min_samples_for_inference=self.interface_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
            max_unique_for_category=self.interface_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
            min_unique_for_numerical=self.interface_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
        )
        # Store the fitted preprocessor (ordinal encoder for text/cats)
        self.preprocessor_ = ord_encoder

        # --- Generate Regressor Ensemble Configs ---
        # (Copied and adapted from the `fit` method)
        possible_target_transforms = (
            ReshapeFeatureDistributionsStep.get_all_preprocessors(
                num_examples=len(y_proc), # Use length of validated y
                random_state=rng, # Use the provided rng
            )
        )
        target_preprocessors: list[TransformerMixin | Pipeline | None] = []
        for (
            y_target_preprocessor
        ) in self.interface_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS:
            if y_target_preprocessor is not None:
                preprocessor = possible_target_transforms[y_target_preprocessor]
            else:
                preprocessor = None
            target_preprocessors.append(preprocessor)

        preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS
        ensemble_configs = EnsembleConfig.generate_for_regression(
            n=self.n_estimators,
            subsample_size=self.interface_config_.SUBSAMPLE_SAMPLES,
            add_fingerprint_feature=self.interface_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.interface_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.interface_config_.POLYNOMIAL_FEATURES,
            max_index=len(X_proc), # Use length of validated X
            preprocessor_configs=typing.cast(
                Sequence[PreprocessorConfig],
                preprocess_transforms
                if preprocess_transforms is not None
                else default_regressor_preprocessor_configs(),
            ),
            target_transforms=target_preprocessors,
            random_state=rng,
        )
        assert len(ensemble_configs) == self.n_estimators

        # Calculate mean/std for *this specific* chunk (for reference, though we use the global ones from self)
        y_mean = np.mean(y_proc).item()
        y_std = np.std(y_proc).item() + 1e-20
        # Standardize y *here* just to return it, though the Dataset uses the global mean/std
        y_standardized = (y_proc - y_mean) / y_std

        # Return the ensemble configs, original X, standardized y (for ref), and stats
        return ensemble_configs, X, y_standardized, y_mean, y_std
    
    def fit_from_preprocessed(
        self,
        X_preprocessed: list[torch.Tensor],
        y_preprocessed: list[torch.Tensor], # These y are standardized
        cat_ix: list[list[int]],
        configs: list[list[EnsembleConfig]], # Should be RegressorEnsembleConfig
        *,
        no_refit=True,
    ) -> TabPFNRegressor:
        """Fit the model using preprocessed inputs from the fine-tuning Dataset.
           Sets the model's context for subsequent prediction on the test split.

        Args:
            X_preprocessed: List (batch size) of lists (ensemble size) of preprocessed
                           training features [EnsembleSize, N_train, Features].
            y_preprocessed: List (batch size) of lists (ensemble size) of preprocessed
                           (standardized) training targets [EnsembleSize, N_train].
            cat_ix: List (batch size) of lists (ensemble size) of categorical feature indices.
            configs: List (batch size) of lists (ensemble size) of EnsembleConfigs used.
            no_refit: If True (default), avoids re-initializing model weights if called multiple times.

        Returns:
            self
        """
        # Initialize model if it hasn't been (e.g., if fine-tuning starts without .fit())
        # This requires some dummy data to infer y stats if not present
        if not hasattr(self, "model_") or not no_refit:
             # If model exists but we want refit, need to re-init y stats
             # We assume y_train_mean/std are correctly set by now either via fit or get_preprocessed_datasets
            if not hasattr(self, 'y_train_mean_') or not hasattr(self, 'y_train_std_'):
                 raise RuntimeError("y_train_mean_ and y_train_std_ must be set before fit_from_preprocessed. Call fit() or get_preprocessed_datasets() first.")
            byte_size, rng = self._initialize_model_variables(X_preprocessed[0][0][:2,:], y_preprocessed[0][0][:2]) # Use dummy data from batch for shape etc.
        else:
            # If model exists and no_refit is True, just get byte_size
            _, _, byte_size = determine_precision(
                self.inference_precision, self.device_
            )
            rng = None # No need for rng if just setting up executor

        # Create the specialized inference engine for batched processing
        self.executor_ = create_inference_engine(
            X_train=X_preprocessed, # Pass the batched preprocessed data
            y_train=y_preprocessed, # Pass the batched preprocessed (standardized) targets
            model=self.model_,
            ensemble_configs=configs, # Pass the configs for the batch
            cat_ix=cat_ix,           # Pass the cat_ix for the batch
            fit_mode="batched",      # Crucial: use the new batched mode
            device_=self.device_,
            rng=rng, # Pass None if model already initialized
            n_jobs=self.n_jobs,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            # Ensure gradients are allowed by default when using this path
            inference_mode=not self.differentiable_input, # False if differentiable needed (prompt tune)
        )

        return self 
    
    def predict_from_preprocessed(self, X_tests: list[torch.Tensor]) -> torch.Tensor:
        """Predict regression target for preprocessed test sets from the fine-tuning Dataset.
           Allows gradients for backpropagation. Returns un-standardized predictions.

        Args:
            X_tests: List (batch size) of lists (ensemble size) of preprocessed
                     test features [EnsembleSize, N_test, Features].

        Returns:
            A tensor of predicted **un-standardized** target values [BatchSize, N_test].
        """
        if not isinstance(self.executor_, InferenceEngineBatchedNoPreprocessing):
            raise ValueError(
                "predict_from_preprocessed can only be called after "
                "fit_from_preprocessed using the batched fit_mode."
            )
        if not hasattr(self, 'y_train_mean_') or not hasattr(self, 'y_train_std_'):
            raise RuntimeError("y_train_mean_ and y_train_std_ must be set before prediction. Call fit() or get_preprocessed_datasets() first.")


        # Ensure torch.inference_mode is OFF to allow gradients
        self.executor_.use_torch_inference_mode(use_inference=False)

        batched_outputs_logits = [] # Store raw logits from each ensemble member across the batch
        batched_borders = []        # Store corresponding borders

        # executor_.iter_outputs yields per ensemble member across the whole batch
        # output shape: [BatchSize, N_test, N_classes]
        # config shape: [BatchSize] containing EnsembleConfig for that member
        for output_logits, ensemble_configs_for_member in self.executor_.iter_outputs(
            X_tests, # Pass the list of test tensors for the batch
            device=self.device_,
            autocast=self.use_autocast_,
        ):
            batched_outputs_logits.append(output_logits)

            # Process borders for each item in the batch for this ensemble member
            member_borders = []
            std_borders = self.bardist_.borders.cpu().numpy() # Standardized borders
            for batch_idx, config in enumerate(ensemble_configs_for_member):
                 assert isinstance(config, RegressorEnsembleConfig)
                 if config.target_transform is None:
                     borders_t = std_borders.copy()
                     logit_cancel_mask = None
                 else:
                     # IMPORTANT: _transform_borders_one returns np.ndarray
                     logit_cancel_mask, descending_borders, borders_t = (
                         _transform_borders_one(
                             std_borders,
                             target_transform=config.target_transform,
                             repair_nan_borders_after_transform=self.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
                         )
                     )
                     if descending_borders:
                         borders_t = borders_t.flip(-1)
                     # Apply logit mask if needed (can modify output_logits directly if careful)
                     if logit_cancel_mask is not None:
                        # Ensure modification happens correctly for the specific batch item
                        #output_logits[batch_idx, :, logit_cancel_mask] = -torch.inf
                        logit_cancel_mask_t = torch.from_numpy(logit_cancel_mask).to(output_logits.device)
                        output_logits[batch_idx, :, logit_cancel_mask_t] = -torch.inf # Use tensor mask

                
                 member_borders.append(torch.as_tensor(borders_t, device=self.device_))
            batched_borders.append(torch.stack(member_borders)) # Shape [BatchSize, N_borders]


        # Stack results across ensemble members: [EnsembleSize, BatchSize, N_test, N_classes]
        all_logits = torch.stack(batched_outputs_logits, dim=0)
        # Stack borders: [EnsembleSize, BatchSize, N_borders]
        all_borders = torch.stack(batched_borders, dim=0)

        # Apply temperature adjustment (optional, less common for regression loss)
        if self.softmax_temperature != 1.0:
             all_logits = all_logits.float() / self.softmax_temperature

        # Translate probabilities across borders for each ensemble member and batch item
        translated_probs_list = []
        target_borders = self.bardist_.borders.to(self.device_) # Standardized target borders
        for ens_idx in range(self.n_estimators):
            # Get the potentially 2D borders for this ensemble member
            frm_batched = all_borders[ens_idx] # Shape [BatchSize, NumBorders]
            
            # --- START SHAPE CORRECTION & VALIDATION ---
            frm_1d_to_use = None
            if frm_batched.ndim == 2:
                batch_size_frm = frm_batched.shape[0]
                if batch_size_frm == 1:
                    # If BatchSize is 1, just take the first (only) row
                    frm_1d_to_use = frm_batched[0]
                else:
                    # If BatchSize > 1, check if all rows are identical.
                    # This ensures the target_transform was effectively the same for all batch items
                    # for this specific ensemble member index.
                    # Note: Using allclose adds a small overhead and assumes gradients are not needed through this check.
                    if torch.allclose(frm_batched, frm_batched[0].unsqueeze(0), atol=1e-6, rtol=1e-5): # Use allclose for float tensors
                        frm_1d_to_use = frm_batched[0] # All rows are the same, use the first one
                    else:
                        # Borders differ across the batch for this ensemble member.
                        # The current utils._cdf and utils._map_to_bucket_ix cannot handle this.
                        raise RuntimeError(
                            f"predict_from_preprocessed detected varying 'frm' borders (shape {frm_batched.shape}) "
                            f"within the batch for ensemble member {ens_idx}. This implies differing target_transforms "
                            f"were used, which is incompatible with the current border translation utility functions. "
                            f"Ensure consistent target_transform usage per ensemble across batch items or modify utils._cdf."
                        )
            elif frm_batched.ndim == 1:
                # This case might occur if batch size was 1 *and* stacking logic somehow returned 1D
                frm_1d_to_use = frm_batched
            else:
                # Should not happen
                raise ValueError(f"Unexpected shape for frm_batched: {frm_batched.shape}")

            # Ensure we have a valid 1D tensor now
            if frm_1d_to_use is None or frm_1d_to_use.ndim != 1:
                raise RuntimeError(f"Failed to obtain valid 1D 'frm' borders for ensemble {ens_idx}")
            # --- END SHAPE CORRECTION & VALIDATION ---

            member_probs = translate_probs_across_borders(
                 all_logits[ens_idx], # [BatchSize, N_test, N_classes]
                 #frm=all_borders[ens_idx], # [BatchSize, N_borders] - broadcasting applies
                 frm=frm_1d_to_use,        # <--- Use the corrected 1D tensor
                 to=target_borders, # [N_borders_target]
            )
            translated_probs_list.append(member_probs)

        # Stack translated probabilities: [EnsembleSize, BatchSize, N_test, N_target_classes]
        stacked_probs = torch.stack(translated_probs_list, dim=0)

        # Average probabilities (or logits before softmax if average_before_softmax is True)
        if self.average_before_softmax:
             # Average log-probs (logits), then apply softmax
             final_logits = torch.log(stacked_probs + 1e-10).mean(dim=0) # Add epsilon
             final_probs = torch.softmax(final_logits, dim=-1)
        else:
             # Average probabilities directly
             final_probs = stacked_probs.mean(dim=0) # Shape [BatchSize, N_test, N_target_classes]


        # Calculate the mean prediction based on the final probabilities and *standardized* criterion
        # Use self.bardist_ which corresponds to the standardized y distribution
        predicted_mean_standardized = self.bardist_.mean(final_probs) # Shape [BatchSize, N_test]

        # --- Crucial: Un-standardize the prediction ---
        predicted_mean_unstandardized = (
            predicted_mean_standardized * self.y_train_std_ + self.y_train_mean_
        )

        # Ensure output shape is appropriate for loss calculation (e.g., [BatchSize, N_test])
        # Might need squeezing if N_test is 1, depending on loss function requirements.
        # Usually, [BatchSize, N_test] or [BatchSize * N_test] is expected.
        # The current shape should be [BatchSize, N_test].

        return predicted_mean_unstandardized

    # --- NEW: Fine-tuning specific methods END ---


    @config_context(transform_output="default")  # type: ignore
    def fit(self, X: XType, y: YType) -> Self:
        """Fit the model.

        Args:
            X: The input data.
            y: The target variable.

        Returns:
            self
        """
        
        byte_size, rng = self._initialize_model_variables(X, y)

        # --- Initialization Phase ---
        # Use helper to initialize model, device, precision, config, y-stats
        byte_size, rng = self._initialize_model_variables(X, y)

        # --- Preprocessing Phase (using helper) ---
        # This validates data, handles categoricals, generates configs
        ensemble_configs, X_processed, y_standardized, _, _ = self._initialize_dataset_preprocessing(
            X, y, rng
        )
        self._ensemble_configs = ensemble_configs # Store configs if needed later

        # --- Inference Engine Setup ---
        # Create the standard inference engine based on self.fit_mode
        # Note: We pass the *standardized* y here for the standard fit modes
        self.executor_ = create_inference_engine(
            X_train=X_processed, # Use the data after categorical processing
            y_train=y_standardized, # Use standardized y
            model=self.model_,
            ensemble_configs=ensemble_configs,
            cat_ix=self.inferred_categorical_indices_,
            fit_mode=self.fit_mode, # Use standard fit modes here
            device_=self.device_,
            rng=rng,
            n_jobs=self.n_jobs,
            byte_size=byte_size,
            forced_inference_dtype_=self.forced_inference_dtype_,
            memory_saving_mode=self.memory_saving_mode,
            use_autocast_=self.use_autocast_,
            # Standard fit usually uses inference_mode=True
            inference_mode=True,
        )

        return self

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["mean", "median", "mode"] = "mean",
        quantiles: list[float] | None = None,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["quantiles"],
        quantiles: list[float] | None = None,
    ) -> list[np.ndarray]: ...

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["main"],
        quantiles: list[float] | None = None,
    ) -> MainOutputDict: ...

    @overload
    def predict(
        self,
        X: XType,
        *,
        output_type: Literal["full"],
        quantiles: list[float] | None = None,
    ) -> FullOutputDict: ...

    # FIXME: improve to not have noqa C901, PLR0912
    @config_context(transform_output="default")  # type: ignore
    def predict(  # noqa: C901, PLR0912
        self,
        X: XType,
        *,
        # TODO: support "ei", "pi"
        output_type: Literal[
            "mean",
            "median",
            "mode",
            "quantiles",
            "full",
            "main",
        ] = "mean",
        quantiles: list[float] | None = None,
    ) -> np.ndarray | list[np.ndarray] | MainOutputDict | FullOutputDict:
        """Predict the target variable.

        Args:
            X: The input data.
            output_type:
                Determines the type of output to return.

                - If `"mean"`, we return the mean over the predicted distribution.
                - If `"median"`, we return the median over the predicted distribution.
                - If `"mode"`, we return the mode over the predicted distribution.
                - If `"quantiles"`, we return the quantiles of the predicted
                    distribution. The parameter `output_quantiles` determines which
                    quantiles are returned.
                - If `"main"`, we return the all output types above in a dict.
                - If `"full"`, we return the full output of the model, including the
                  logits and the criterion, and all the output types from "main".

            quantiles:
                The quantiles to return if `output="quantiles"`.

                By default, the `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
                quantiles are returned. The predictions per quantile match
                the input order.

        Returns:
            The predicted target variable or a list of predictions per quantile.
        """
        check_is_fitted(self)

        X = validate_X_predict(X, self)
        X = _fix_dtypes(X, cat_indices=self.categorical_features_indices)
        X = _process_text_na_dataframe(X, ord_encoder=self.preprocessor_)  # type: ignore

        # --- Standard Prediction Logic ---
        # (This part is largely the same as the original code)
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        else:
            assert all(
                (0 <= q <= 1) and (isinstance(q, float)) for q in quantiles
            ), "All quantiles must be between 0 and 1 and floats."
        if output_type not in self._USABLE_OUTPUT_TYPES:
            raise ValueError(f"Invalid output type: {output_type}")

        std_borders = self.bardist_.borders.cpu().numpy()
        outputs: list[torch.Tensor] = []
        borders: list[np.ndarray] = []

        for output, config in self.executor_.iter_outputs(
            X,
            device=self.device_,
            autocast=self.use_autocast_,
        ):
            assert isinstance(config, RegressorEnsembleConfig)

            if self.softmax_temperature != 1:
                output = output.float() / self.softmax_temperature  # noqa: PLW2901

            borders_t: np.ndarray
            logit_cancel_mask: np.ndarray | None
            descending_borders: bool

            # TODO(eddiebergman): Maybe this could be parallelized or done in fit
            # but I somehow doubt it takes much time to be worth it.
            # One reason to make it worth it is if you want fast predictions, i.e.
            # don't re-do this each time.
            # However it gets a bit more difficult as you need to line up the
            # outputs from `iter_outputs` above (which may be in arbitrary order),
            # along with the specific config the output belongs to. This is because
            # the transformation done to the borders for a given output is dependant
            # upon the target_transform of the config.
            if config.target_transform is None:
                borders_t = std_borders.copy()
                logit_cancel_mask = None
                descending_borders = False
            else:
                logit_cancel_mask, descending_borders, borders_t = (
                    _transform_borders_one(
                        std_borders,
                        target_transform=config.target_transform,
                        repair_nan_borders_after_transform=self.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
                    )
                )
                if descending_borders:
                    borders_t = borders_t.flip(-1)  # type: ignore

            borders.append(borders_t)

            if logit_cancel_mask is not None:
                output = output.clone()  # noqa: PLW2901
                output[..., logit_cancel_mask] = float("-inf")

            outputs.append(output)  # type: ignore

        # --- Translate probs, average, get final logits (same as before) ---
        transformed_probs = [ # Use probs here not logits for translation
            translate_probs_across_borders(
                torch.softmax(logits, dim=-1), # Convert logits to probs first
                frm=torch.as_tensor(borders_t, device=self.device_),
                to=self.bardist_.borders.to(self.device_),
            )
            for logits, borders_t in zip(outputs, borders)
        ]
        stacked_probs = torch.stack(transformed_probs, dim=0) # [Ensemble, N_samples, N_classes]

        if self.average_before_softmax:
            # Need to average the *original* logits before softmax
            original_logits = torch.stack(outputs, dim=0) # [Ensemble, N_samples, N_classes]
            avg_logits = original_logits.mean(dim=0)
            # Now translate these averaged logits
            final_probs = translate_probs_across_borders(
                 torch.softmax(avg_logits, dim=-1), # Probs from averaged logits
                 frm=torch.as_tensor(np.mean(borders, axis=0), device=self.device_), # Use average border? Or need per-sample? Simpler: Use standardized target borders.
                 to=self.bardist_.borders.to(self.device_), # Standardized target borders
             )
        else:
            # Average probabilities after translation
            final_probs = stacked_probs.mean(dim=0) # [N_samples, N_classes]


        # Use final_probs to calculate outputs
        logits = torch.log(final_probs + 1e-10) # Convert back to log-probs for criterion functions
        if logits.dtype == torch.float16:
            logits = logits.float()
        logits = logits.cpu()

        # Determine and return intended output type
        logit_to_output = partial(
            _logits_to_output,
            logits=logits,
            criterion=self.renormalized_criterion_,
            quantiles=quantiles,
        )
        if output_type in ["full", "main"]:
            # Create a dictionary of outputs with proper typing via TypedDict
            # Get individual outputs with proper typing
            mean_out = typing.cast(np.ndarray, logit_to_output(output_type="mean"))
            median_out = typing.cast(np.ndarray, logit_to_output(output_type="median"))
            mode_out = typing.cast(np.ndarray, logit_to_output(output_type="mode"))
            quantiles_out = typing.cast(
                list[np.ndarray],
                logit_to_output(output_type="quantiles"),
            )

            # Create our typed dictionary
            main_outputs = MainOutputDict(
                mean=mean_out,
                median=median_out,
                mode=mode_out,
                quantiles=quantiles_out,
            )

            if output_type == "full":
                # Return full output with criterion and logits
                return FullOutputDict(
                    **main_outputs,
                    criterion=self.renormalized_criterion_,
                    logits=logits,
                )

            return main_outputs
        return logit_to_output(output_type=output_type)

    def get_embeddings(
        self,
        X: XType,
        data_source: Literal["train", "test"] = "test",
    ) -> np.ndarray:
        """Get the embeddings for the input data `X`.

        Parameters:
            X (XType): The input data.
            data_source str: Extract either the train or test embeddings
        Returns:
            np.ndarray: The computed embeddings for each fitted estimator.
        """
        return _get_embeddings(self, X, data_source)


def _logits_to_output(
    *,
    output_type: str,
    logits: torch.Tensor,
    criterion: FullSupportBarDistribution,
    quantiles: list[float],
) -> np.ndarray | list[np.ndarray]:
    """Convert the logits to the specified output type.

    Args:
        output_type: The output type to convert the logits to.
        logits: The logits to convert.
        criterion: The criterion to use for the conversion.
        quantiles: The quantiles to use for the conversion.

    Returns:
        The converted logits or list of converted logits.
    """
    if output_type == "quantiles":
        return [criterion.icdf(logits, q).cpu().detach().numpy() for q in quantiles]

    # TODO: support
    #   "pi": criterion.pi(logits, np.max(self.y)), # noqa: ERA001
    #   "ei": criterion.ei(logits), # noqa: ERA001
    if output_type == "mean":
        output = criterion.mean(logits)
    elif output_type == "median":
        output = criterion.median(logits)
    elif output_type == "mode":
        output = criterion.mode(logits)
    else:
        raise ValueError(f"Invalid output type: {output_type}")

    return output.cpu().detach().numpy()  # type: ignore
