import os
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import load_digits

from aiforge.core.config import ExperimentConfig
from aiforge.data.dataset import DataEngine


def test_digits_builtin_no_val_test_leakage():
    """
    Outliers injected exclusively into validation and test splits of digits_builtin
    must NOT alter the fitted StandardScaler mean or scale.
    """
    cfg = ExperimentConfig(
        seed=42,
        data={"type": "digits_builtin"},
        architecture={
            "type": "mlp",
            "mlp_config": {"input_size": 64, "hidden_sizes": [16], "output_size": 10},
        },
    )

    _, _, _, baseline_artifacts = DataEngine.get_dataloaders(cfg)
    baseline_mean = np.array(baseline_artifacts["scaler"]["mean"])
    baseline_scale = np.array(baseline_artifacts["scaler"]["scale"])

    # Determine the exact train/val/test indices generated with seed=42
    raw_digits = load_digits()
    total_size = len(raw_digits.data)
    train_size = int(0.7 * total_size)
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(total_size, generator=gen).numpy()
    non_train_idx = perm[train_size:]

    corrupted_digits = load_digits()
    corrupted_digits.data = corrupted_digits.data.astype(np.float32)
    corrupted_digits.data[non_train_idx] = 1e9  # massive outlier in val + test only

    with patch("aiforge.data.dataset.load_digits", return_value=corrupted_digits):
        _, _, _, corrupted_artifacts = DataEngine.get_dataloaders(cfg)

    np.testing.assert_allclose(
        corrupted_artifacts["scaler"]["mean"], baseline_mean, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        corrupted_artifacts["scaler"]["scale"], baseline_scale, rtol=1e-6, atol=1e-6
    )


def test_csv_no_val_test_leakage_in_imputation_and_scaler(tmp_path):
    """
    Outliers and NaNs in val/test splits of a CSV dataset must NOT alter
    imputation means or StandardScaler statistics fitted on train.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(123)
    n_rows = 100
    f1 = rng.normal(loc=10.0, scale=2.0, size=n_rows).astype(np.float32)
    f2 = rng.normal(loc=-5.0, scale=1.0, size=n_rows).astype(np.float32)
    target = rng.integers(0, 2, size=n_rows)

    # Inject a couple of NaNs in train so mean imputation is actively exercised
    idx_all = np.arange(n_rows)
    idx_temp, idx_test = train_test_split(idx_all, test_size=0.15, random_state=99)
    idx_train, idx_val = train_test_split(idx_temp, test_size=0.1765, random_state=99)

    f1[idx_train[0]] = np.nan
    f2[idx_train[1]] = np.nan

    csv_clean = tmp_path / "clean.csv"
    pd.DataFrame({"f1": f1, "f2": f2, "label": target}).to_csv(csv_clean, index=False)

    cfg_clean = ExperimentConfig(
        seed=99,
        data={
            "type": "csv",
            "csv_config": {
                "path": str(csv_clean),
                "target_column": "label",
                "missing_values": "mean",
            },
        },
        architecture={
            "type": "mlp",
            "mlp_config": {"input_size": 2, "hidden_sizes": [8], "output_size": 2},
        },
    )
    _, _, _, art_clean = DataEngine.get_dataloaders(cfg_clean)

    # Now corrupt ONLY val and test rows with extreme outliers and NaNs
    f1_corrupt = f1.copy()
    f2_corrupt = f2.copy()
    f1_corrupt[idx_val] = 1e8
    f2_corrupt[idx_test] = -1e8
    f1_corrupt[idx_test[0]] = np.nan

    csv_corrupt = tmp_path / "corrupt.csv"
    pd.DataFrame({"f1": f1_corrupt, "f2": f2_corrupt, "label": target}).to_csv(
        csv_corrupt, index=False
    )

    cfg_corrupt = ExperimentConfig(
        seed=99,
        data={
            "type": "csv",
            "csv_config": {
                "path": str(csv_corrupt),
                "target_column": "label",
                "missing_values": "mean",
            },
        },
        architecture={
            "type": "mlp",
            "mlp_config": {"input_size": 2, "hidden_sizes": [8], "output_size": 2},
        },
    )
    _, val_loader, test_loader, art_corrupt = DataEngine.get_dataloaders(cfg_corrupt)

    np.testing.assert_allclose(
        art_corrupt["imputation"]["means"],
        art_clean["imputation"]["means"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        art_corrupt["scaler"]["mean"],
        art_clean["scaler"]["mean"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        art_corrupt["scaler"]["scale"],
        art_clean["scaler"]["scale"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert art_corrupt["feature_columns"] == ["f1", "f2"]

    # Ensure NaNs in test split were imputed using train mean (no NaNs in tensors)
    for x_batch, _ in test_loader:
        assert not torch.isnan(x_batch).any()


def test_csv_all_nan_training_column_rejected(tmp_path):
    """
    If a feature column has only NaN values in the training split,
    DataEngine must raise an explicit ValueError instead of propagating NaNs.
    """
    csv_path = tmp_path / "all_nan.csv"
    df = pd.DataFrame(
        {
            "f_valid": np.arange(30, dtype=np.float32),
            "f_all_nan": [np.nan] * 30,
            "target": [0, 1] * 15,
        }
    )
    df.to_csv(csv_path, index=False)

    cfg = ExperimentConfig(
        seed=7,
        data={
            "type": "csv",
            "csv_config": {
                "path": str(csv_path),
                "target_column": "target",
                "missing_values": "mean",
            },
        },
        architecture={
            "type": "mlp",
            "mlp_config": {"input_size": 2, "hidden_sizes": [8], "output_size": 2},
        },
    )

    with pytest.raises(ValueError, match="f_all_nan"):
        DataEngine.get_dataloaders(cfg)


def test_saved_artifacts_reused_without_refitting(tmp_path):
    """
    When saved_artifacts are provided, DataEngine must reuse saved feature_columns,
    imputation means, and scaler statistics without refitting.
    """
    csv_path = tmp_path / "reuse.csv"
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
            "b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] * 3,
            "target": [0, 1] * 15,
        }
    )
    df.to_csv(csv_path, index=False)

    cfg = ExperimentConfig(
        seed=42,
        data={
            "type": "csv",
            "csv_config": {
                "path": str(csv_path),
                "target_column": "target",
                "missing_values": "mean",
            },
        },
        architecture={
            "type": "mlp",
            "mlp_config": {"input_size": 2, "hidden_sizes": [8], "output_size": 2},
        },
    )

    custom_artifacts = {
        "feature_columns": ["a", "b"],
        "imputation": {"strategy": "mean", "means": [999.0, -999.0]},
        "scaler": {"type": "StandardScaler", "mean": [0.0, 0.0], "scale": [1.0, 1.0]},
    }

    train_loader, _, _, returned_artifacts = DataEngine.get_dataloaders(
        cfg, saved_artifacts=custom_artifacts
    )
    assert returned_artifacts["imputation"]["means"] == [999.0, -999.0]
    assert returned_artifacts["scaler"]["mean"] == [0.0, 0.0]

    all_train_x = torch.cat([x for x, _ in train_loader], dim=0).numpy()
    assert np.isclose(all_train_x[:, 0], 999.0).any()
