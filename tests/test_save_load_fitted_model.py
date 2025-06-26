from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_regression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabpfn import TabPFNRegressor
from tabpfn.model.loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
    preds_before = reg.predict(X[:5])
    save_fitted_tabpfn_model(reg, path)
    loaded = load_fitted_tabpfn_model(path, device="cpu")

    np.testing.assert_allclose(preds_before, loaded.predict(X[:5]))
)


def test_save_and_load_fitted_model(tmp_path):
    X, y = make_regression(n_samples=20, n_features=4, random_state=0)
    reg = TabPFNRegressor(
        n_estimators=1,
        device="cpu",
        fit_mode="fit_preprocessors",
        inference_precision=torch.float32,
    )
    reg.fit(X, y)

    path = tmp_path / "model.tabpfn_fit"
    preds_before = reg.predict(X[:5])

    save_fitted_tabpfn_model(reg, path)

    loaded = load_fitted_tabpfn_model(path, device="cpu")

    np.testing.assert_allclose(preds_before, loaded.predict(X[:5]))
