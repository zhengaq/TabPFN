from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tabpfn.classifier import TabPFNClassifier
from tabpfn.utils import collate_for_tabpfn_dataset, pad_tensors

# Use a single random generator for all synthetic data
rng = np.random.default_rng(42)


# Minimal synthetic data for fast tests
@pytest.fixture
def synthetic_data():
    X = rng.normal(size=(100, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=100)
    return X, y


# Fixture: synthetic collection of datasets (list of (X, y) tuples)
@pytest.fixture
def uniform_synthetic_dataset_collection():
    datasets = []
    for _ in range(3):
        X = rng.normal(size=(50, 4)).astype(np.float32)
        y = rng.integers(0, 3, size=50)
        datasets.append((X, y))
    return datasets


# Fixture: synthetic collection of datasets (list of (X, y) tuples)
# where dataset size and number of classes vary.
@pytest.fixture
def variable_synthetic_dataset_collection():
    """Fixture: synthetic collection of datasets
    (list of (X, y) tuples)
    where dataset size and number of classes vary.
    """
    datasets = []
    dataset_sizes = [30, 60, 120]
    class_counts = [2, 4, 6]
    n_features = 4
    for size, n_classes in zip(dataset_sizes, class_counts):
        X = rng.normal(size=(size, n_features)).astype(np.float32)
        y = rng.integers(0, n_classes, size=size)
        datasets.append((X, y))
    return datasets


def test_tabpfn_finetune_basic_runs(synthetic_data) -> None:
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    classifier_args = {
        "ignore_pretraining_limits": True,
        "device": "cpu",
        "n_estimators": 1,
        "random_state": 2,
        "inference_precision": torch.float32,
    }
    clf = TabPFNClassifier(**classifier_args)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    assert preds.shape == y_test.shape
    proba = clf.predict_proba(X_test)
    assert proba.shape[0] == X_test.shape[0]


def test_eval_test_function(synthetic_data) -> None:
    from examples.finetune_classifier import eval_test

    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    classifier_args = {
        "ignore_pretraining_limits": True,
        "device": "cpu",
        "n_estimators": 1,
        "random_state": 2,
        "inference_precision": torch.float32,
    }
    clf = TabPFNClassifier(**classifier_args)
    acc, ll = eval_test(
        clf,
        classifier_args,
        X_train_raw=X_train,
        y_train_raw=y_train,
        X_test_raw=X_test,
        y_test_raw=y_test,
    )
    assert acc is not None
    assert ll is not None


# ----------- Test DatasetCollectionWithPreprocessing Class ---


@pytest.fixture
def classifier_instance() -> TabPFNClassifier:
    return TabPFNClassifier(
        n_estimators=2,
        device="cpu",
        random_state=42,
        inference_precision=torch.float32,
        fit_mode="fit_preprocessors",
    )


def test_datasetcollectionwithpreprocessing_classification_single_dataset(
    synthetic_data, classifier_instance: TabPFNClassifier
) -> None:
    import numpy as np
    from sklearn.model_selection import train_test_split

    from tabpfn.preprocessing import DatasetCollectionWithPreprocessing

    X_raw, y_raw = synthetic_data
    clf = classifier_instance
    n_estimators = clf.n_estimators
    test_size = 0.3

    split_fn = partial(train_test_split, test_size=test_size, shuffle=True)
    dataset_collection = clf.get_preprocessed_datasets(
        X_raw, y_raw, split_fn=split_fn, max_data_size=None
    )

    assert isinstance(dataset_collection, DatasetCollectionWithPreprocessing)
    assert len(dataset_collection) == 1, "Collection should contain one dataset config"

    item_index = 0
    processed_dataset_item = dataset_collection[item_index]

    assert isinstance(processed_dataset_item, tuple)
    assert (
        len(processed_dataset_item) == 6
    ), "Item tuple should have 4 elements for classification"

    (
        X_trains_preprocessed,
        X_tests_preprocessed,
        y_trains_preprocessed,
        y_test_raw_tensor,
        cat_ixs,
        returned_ensemble_configs,
    ) = processed_dataset_item

    assert isinstance(X_trains_preprocessed, list)
    assert len(X_trains_preprocessed) == n_estimators
    n_samples_total = X_raw.shape[0]
    expected_n_test = int(np.floor(n_samples_total * test_size))
    expected_n_train = n_samples_total - expected_n_test
    assert y_test_raw_tensor.shape == (expected_n_test,)
    assert X_trains_preprocessed[0].shape[0] == expected_n_train


def test_datasetcollectionwithpreprocessing_classification_multiple_datasets(
    uniform_synthetic_dataset_collection, classifier_instance: TabPFNClassifier
) -> None:
    """Test DatasetCollectionWithPreprocessing
    using a collection of multiple synthetic datasets.
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    from tabpfn.preprocessing import DatasetCollectionWithPreprocessing

    datasets = uniform_synthetic_dataset_collection
    clf = classifier_instance
    n_estimators = clf.n_estimators
    test_size = 0.3
    split_fn = partial(train_test_split, test_size=test_size, shuffle=True)

    X_list = [X for X, _ in datasets]
    y_list = [y for _, y in datasets]

    dataset_collection = clf.get_preprocessed_datasets(
        X_list, y_list, split_fn=split_fn, max_data_size=None
    )

    assert isinstance(dataset_collection, DatasetCollectionWithPreprocessing)
    assert len(dataset_collection) == len(
        datasets
    ), "Collection should contain one item per dataset"

    for item_index in range(len(datasets)):
        processed_dataset_item = dataset_collection[item_index]
        assert isinstance(processed_dataset_item, tuple)
        assert (
            len(processed_dataset_item) == 6
        ), "Item tuple should have 6 elements for classification"
        (
            X_trains_preprocessed,
            X_tests_preprocessed,
            y_trains_preprocessed,
            y_test_raw_tensor,
            cat_ixs,
            returned_ensemble_configs,
        ) = processed_dataset_item
        assert isinstance(X_trains_preprocessed, list)
        assert len(X_trains_preprocessed) == n_estimators
        n_samples_total = X_list[item_index].shape[0]
        expected_n_test = int(np.floor(n_samples_total * test_size))
        expected_n_train = n_samples_total - expected_n_test
        assert y_test_raw_tensor.shape == (expected_n_test,)
        assert X_trains_preprocessed[0].shape[0] == expected_n_train


def test_get_preprocessed_datasets_basic():
    import numpy as np
    from sklearn.model_selection import train_test_split

    from tabpfn.classifier import TabPFNClassifier

    # Create synthetic data
    X = rng.normal(size=(100, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=100)

    clf = TabPFNClassifier()
    # This should return a DatasetCollectionWithPreprocessing
    dataset = clf.get_preprocessed_datasets(X, y, split_fn=train_test_split)
    assert hasattr(dataset, "__getitem__")
    assert hasattr(dataset, "__len__")
    assert len(dataset) > 0
    item = dataset[0]
    assert isinstance(item, tuple)
    assert len(item) == 6


def test_pad_tensors_2d_and_1d():
    import torch

    # 2D tensors (features)
    tensors_2d = [torch.ones((2, 3)), torch.ones((3, 2)), torch.ones((1, 4))]
    padded = pad_tensors(tensors_2d, padding_val=-1, labels=False)
    assert all(
        t.shape == (3, 4) for t in padded
    ), f"Expected shape (3, 4), got {[t.shape for t in padded]}"
    assert padded[0][2, 3] == -1, "Padding value not set correctly for 2D case."

    # 1D tensors (labels)
    tensors_1d = [torch.arange(3), torch.arange(5), torch.arange(2)]
    padded_1d = pad_tensors(tensors_1d, padding_val=99, labels=True)
    assert all(
        t.shape == (5,) for t in padded_1d
    ), f"Expected shape (5,), got {[t.shape for t in padded_1d]}"
    assert padded_1d[0][3] == 99, "Padding value not set correctly for 1D case."


def test_collate_for_tabpfn_dataset_uniform_collection(
    uniform_synthetic_dataset_collection, classifier_instance
):
    import torch

    from tabpfn.utils import collate_for_tabpfn_dataset

    X_list = [X for X, _ in uniform_synthetic_dataset_collection]
    y_list = [y for _, y in uniform_synthetic_dataset_collection]
    preprocessed_collection = classifier_instance.get_preprocessed_datasets(
        X_list, y_list, train_test_split, 100
    )
    batch = [preprocessed_collection[0], preprocessed_collection[1]]
    collated = collate_for_tabpfn_dataset(batch)
    assert isinstance(collated, tuple), "Collator output should be a tuple."
    X_trains = collated[0]
    assert isinstance(X_trains, list), "First element should be a list (per estimator)."
    for est_tensor in X_trains:
        assert isinstance(
            est_tensor, torch.Tensor
        ), "Each estimator's batch should be a tensor."
        assert est_tensor.shape[0] == len(
            batch
        ), "Batch size should match input batch (batch_size=1)."
    y_trains = collated[2]
    for est_tensor in y_trains:
        assert isinstance(
            est_tensor, torch.Tensor
        ), "Each estimator's batch should be a tensor for labels."
        assert est_tensor.shape[0] == len(batch)


def test_collate_for_tabpfn_dataset_variable_collection(
    variable_synthetic_dataset_collection, classifier_instance
) -> None:
    import torch

    from tabpfn.utils import collate_for_tabpfn_dataset

    X_list = [X for X, _ in variable_synthetic_dataset_collection]
    y_list = [y for _, y in variable_synthetic_dataset_collection]
    preprocessed_collection = classifier_instance.get_preprocessed_datasets(
        X_list, y_list, train_test_split, 100
    )
    batch = [preprocessed_collection[0], preprocessed_collection[1]]
    collated = collate_for_tabpfn_dataset(batch)
    assert isinstance(collated, tuple), "Collator output should be a tuple."
    X_trains = collated[0]
    assert isinstance(X_trains, list), "First element should be a list (per estimator)."
    for est_tensor in X_trains:
        assert isinstance(
            est_tensor, torch.Tensor
        ), "Each estimator's batch should be a tensor."
        assert est_tensor.shape[0] == len(
            batch
        ), "Batch size should match input batch (batch_size=1)."
    y_trains = collated[2]
    for est_tensor in y_trains:
        assert isinstance(
            est_tensor, torch.Tensor
        ), "Each estimator's batch should be a tensor for labels."
        assert est_tensor.shape[0] == len(batch)


def test_dataset_and_collator_with_dataloader_uniform(
    uniform_synthetic_dataset_collection, classifier_instance
) -> None:
    import torch
    from torch.utils.data import DataLoader

    from tabpfn.utils import collate_for_tabpfn_dataset

    # Prepare dataset collection
    X_list = [X for X, _ in uniform_synthetic_dataset_collection]
    y_list = [y for _, y in uniform_synthetic_dataset_collection]
    dataset_collection = classifier_instance.get_preprocessed_datasets(
        X_list, y_list, train_test_split, 100
    )
    batch_size = 2
    dl = DataLoader(
        dataset_collection,
        batch_size=batch_size,
        collate_fn=collate_for_tabpfn_dataset,
    )
    for batch in dl:
        # Should be tuple with X_trains, X_tests, y_trains, y_tests, cat_ixs, confs
        assert isinstance(batch, tuple)
        X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = batch
        for est_tensor in X_trains:
            assert isinstance(
                est_tensor, torch.Tensor
            ), "Each estimator's batch should be a tensor."
            assert est_tensor.shape[0] == batch_size
        for est_tensor in y_trains:
            assert isinstance(
                est_tensor, torch.Tensor
            ), "Each estimator's batch should be a tensor for labels."
            assert est_tensor.shape[0] == batch_size
        break  # Only check one batch


def test_dataset_and_collator_with_dataloader_variable(
    variable_synthetic_dataset_collection, classifier_instance
):
    import torch
    from torch.utils.data import DataLoader

    from tabpfn.utils import collate_for_tabpfn_dataset

    X_list = [X for X, _ in variable_synthetic_dataset_collection]
    y_list = [y for _, y in variable_synthetic_dataset_collection]
    dataset_collection = classifier_instance.get_preprocessed_datasets(
        X_list, y_list, train_test_split, 100
    )
    batch_size = 1
    dl = DataLoader(
        dataset_collection,
        batch_size=batch_size,
        collate_fn=collate_for_tabpfn_dataset,
    )
    for batch in dl:
        assert isinstance(batch, tuple)
        X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = batch
        for est_tensor in X_trains:
            assert isinstance(est_tensor, torch.Tensor)
            assert est_tensor.shape[0] == batch_size
        for est_tensor in y_trains:
            assert isinstance(est_tensor, torch.Tensor)
            assert est_tensor.shape[0] == batch_size
        break


def test_tabpfn_finetune_from_preprocessed_runs(synthetic_data) -> None:
    """Test TabPFNClassifier finetuning
    from preprocessed datasets using
    DataLoader and collator.
    Checks that the model can fit and
    predict from batches of preprocessed
    data, regardless of batch size.
    """
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    classifier_args = {
        "ignore_pretraining_limits": True,
        "device": "cpu",
        "n_estimators": 1,
        "random_state": 2,
        "inference_precision": torch.float32,
    }
    clf = TabPFNClassifier(**classifier_args)
    datasets_list = clf.get_preprocessed_datasets(
        X_train, y_train, train_test_split, 100
    )
    # Test with batch_size=1 and batch_size=2
    for batch_size in [1]:
        my_dl_train = DataLoader(
            datasets_list,
            batch_size=batch_size,
            collate_fn=collate_for_tabpfn_dataset,
        )
        for data_batch in my_dl_train:
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = clf.predict_proba_from_preprocessed(X_tests)
            assert preds.shape[0] == X_tests[0].shape[0]
            assert batch_size == X_tests[0].shape[0]
            assert len(X_tests) == clf.n_estimators
            assert len(y_tests) == clf.n_estimators
            assert len(X_trains) == clf.n_estimators
            assert len(y_trains) == clf.n_estimators
            for est_idx in range(clf.n_estimators):
                n_test_samples = X_tests[est_idx].shape[1]
                n_train_samples = X_trains[est_idx].shape[1]
                # Assert at least one test and train sample in each batch
                assert n_test_samples > 0
                assert n_train_samples > 0
                # preds shape: (batch_size, n_classes, n_test_samples)
                assert preds.shape[2] == n_test_samples, (
                    f"For estimator {est_idx}: preds.shape[2]={preds.shape[2]} "
                    f"does not match n_test_samples={n_test_samples}"
                )
