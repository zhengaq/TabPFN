from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_regression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabpfn import TabPFNRegressor


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
    reg.save_fit_state(path)

    loaded = TabPFNRegressor.load_from_fit_state(path, device="cpu")

    np.testing.assert_allclose(reg.predict(X[:5]), loaded.predict(X[:5]))
