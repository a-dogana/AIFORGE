import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional
from aiforge.core.config import ExperimentConfig

class DigitsDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class CSVDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class DataEngine:
    @staticmethod
    def get_dataloaders(config: ExperimentConfig, saved_artifacts: Optional[Dict[str, Any]] = None) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
        """
        Returns train, val, test loaders and a dictionary of preprocessing artifacts.
        If saved_artifacts is provided, uses them instead of fitting new ones.
        """
        data_type = config.data.type
        artifacts = {}
        if saved_artifacts is None:
            saved_artifacts = {}
        
        # Generator for reproducible splits
        generator = torch.Generator().manual_seed(config.seed) if config.seed is not None else None

        if data_type == "digits_builtin":
            digits = load_digits()
            X = digits.data.astype(np.float32)
            y = digits.target
            
            total_size = len(X)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            if generator is not None:
                perm = torch.randperm(total_size, generator=generator).numpy()
            else:
                perm = np.random.permutation(total_size)

            train_idx = perm[:train_size]
            val_idx = perm[train_size : train_size + val_size]
            test_idx = perm[train_size + val_size :]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            scaler = StandardScaler()
            if "scaler" in saved_artifacts:
                scaler.mean_ = np.array(saved_artifacts["scaler"]["mean"], dtype=np.float64)
                scaler.scale_ = np.array(saved_artifacts["scaler"]["scale"], dtype=np.float64)
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_features_in_ = X_train.shape[1]
                X_train = scaler.transform(X_train)
                artifacts["scaler"] = saved_artifacts["scaler"]
            else:
                X_train = scaler.fit_transform(X_train)
                artifacts["scaler"] = {
                    "type": "StandardScaler",
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                }

            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            artifacts["feature_columns"] = [f"pixel_{i}" for i in range(X.shape[1])]

            train_dataset = DigitsDataset(X_train, y_train)
            val_dataset = DigitsDataset(X_val, y_val)
            test_dataset = DigitsDataset(X_test, y_test)
            
        elif data_type == "csv":
            cfg = config.data.csv_config
            if not os.path.exists(cfg.path):
                raise FileNotFoundError(f"CSV file not found: {cfg.path}")
                
            df = pd.read_csv(cfg.path)
            
            if cfg.target_column not in df.columns:
                raise ValueError(f"Target column '{cfg.target_column}' not found in CSV.")
                
            # Drop excluded columns
            drop_cols = [c for c in cfg.exclude_columns if c in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

            # Drop rows missing target column first
            df = df.dropna(subset=[cfg.target_column])
            if cfg.missing_values == "drop":
                df = df.dropna()
            if len(df) == 0:
                raise ValueError("No valid rows remaining in CSV after dropping missing values.")

            if "feature_columns" in saved_artifacts and saved_artifacts["feature_columns"]:
                feature_columns = list(saved_artifacts["feature_columns"])
                missing_feats = [c for c in feature_columns if c not in df.columns]
                if missing_feats:
                    raise ValueError(f"Saved feature columns missing from CSV: {missing_feats}")
                X_df = df[feature_columns]
            else:
                X_df = df.drop(columns=[cfg.target_column])
                feature_columns = list(X_df.columns)

            y = df[cfg.target_column].values
            
            # Convert features to float32 (preserving NaN for train-only imputation)
            try:
                X = X_df.values.astype(np.float32)
            except ValueError as e:
                raise ValueError("Could not convert CSV features to float32. Categorical encoding not fully supported yet.") from e
                
            # Split BEFORE fitting imputation means and StandardScaler to prevent data leakage
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=config.seed)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=config.seed) # ~15% of total

            if cfg.missing_values == "mean":
                if "imputation" in saved_artifacts and "means" in saved_artifacts["imputation"]:
                    col_means = np.array(saved_artifacts["imputation"]["means"], dtype=np.float32)
                    artifacts["imputation"] = saved_artifacts["imputation"]
                else:
                    with np.errstate(invalid="ignore"):
                        col_means = np.nanmean(X_train, axis=0).astype(np.float32)
                    nan_cols = [feature_columns[i] for i, m in enumerate(col_means) if np.isnan(m)]
                    if nan_cols:
                        raise ValueError(
                            f"Feature column(s) {nan_cols} contain only NaN values in the training split; cannot compute imputation mean."
                        )
                    artifacts["imputation"] = {
                        "strategy": "mean",
                        "means": col_means.tolist(),
                    }

                def _impute_with_means(arr: np.ndarray, means: np.ndarray) -> np.ndarray:
                    out = arr.copy()
                    nan_mask = np.isnan(out)
                    if nan_mask.any():
                        out[nan_mask] = np.take(means, np.where(nan_mask)[1])
                    return out

                X_train = _impute_with_means(X_train, col_means)
                X_val = _impute_with_means(X_val, col_means)
                X_test = _impute_with_means(X_test, col_means)
            else:
                artifacts["imputation"] = {"strategy": cfg.missing_values}
            
            scaler = StandardScaler()
            if "scaler" in saved_artifacts:
                scaler.mean_ = np.array(saved_artifacts["scaler"]["mean"], dtype=np.float64)
                scaler.scale_ = np.array(saved_artifacts["scaler"]["scale"], dtype=np.float64)
                scaler.var_ = scaler.scale_ ** 2
                scaler.n_features_in_ = X_train.shape[1]
                
                X_train = scaler.transform(X_train)
                artifacts["scaler"] = saved_artifacts["scaler"]
            else:
                X_train = scaler.fit_transform(X_train)
                artifacts["scaler"] = {
                    "type": "StandardScaler",
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist()
                }
                
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            artifacts["feature_columns"] = feature_columns
            
            train_dataset = CSVDataset(X_train, y_train)
            val_dataset = CSVDataset(X_val, y_val)
            test_dataset = CSVDataset(X_test, y_test)
            
        elif data_type == "images":
            cfg = config.data.image_config
            if not os.path.exists(cfg.path):
                raise FileNotFoundError(f"Image directory not found: {cfg.path}")
                
            transform = transforms.Compose([
                transforms.Resize((cfg.image_size[0], cfg.image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet
            ])
            artifacts["transform"] = "from_config"
            
            if cfg.mode == "single":
                full_dataset = datasets.ImageFolder(root=cfg.path, transform=transform)
                total_size = len(full_dataset)
                train_size = int(0.7 * total_size)
                val_size = int(0.15 * total_size)
                test_size = total_size - train_size - val_size
                
                train_dataset, val_dataset, test_dataset = random_split(
                    full_dataset, [train_size, val_size, test_size], generator=generator
                )
            elif cfg.mode == "folders":
                train_path = os.path.join(cfg.path, "train")
                val_path = os.path.join(cfg.path, "val")
                # Fallback test to val if not exists
                test_path = os.path.join(cfg.path, "test") if os.path.exists(os.path.join(cfg.path, "test")) else val_path
                
                if not os.path.exists(train_path) or not os.path.exists(val_path):
                    raise FileNotFoundError("Image mode 'folders' requires 'train' and 'val' subdirectories.")
                    
                train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
                val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
                test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
            else:
                 raise ValueError(f"Unknown image mode: {cfg.mode}")
        else:
            raise ValueError(f"Unknown data type: {data_type}")
            
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, artifacts
