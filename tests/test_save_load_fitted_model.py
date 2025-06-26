from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_regression

from tabpfn import TabPFNRegressor
from tabpfn.model.loading import load_fitted_tabpfn_model, save_fitted_tabpfn_model


def test_save_and_load_fitted_model(tmp_path):
    X, y = make_regression(n_samples=20, n_features=4, random_state=0)
    reg = TabPFNRegressor(
        n_estimators=1,
        device="cpu",
        fit_mode="fit_preprocessors",
        inference_precision=torch.float32,
    )
    reg.fit(X, y)

    path = tmp_path / "model.pkl"
    save_fitted_tabpfn_model(reg, path)

    loaded = load_fitted_tabpfn_model(path, device="cpu")

    np.testing.assert_allclose(reg.predict(X[:5]), loaded.predict(X[:5]))
