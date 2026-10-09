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
            X = digits.data
            y = digits.target
            
            scaler = StandardScaler()
            if "scaler" in saved_artifacts:
                scaler.mean_ = np.array(saved_artifacts["scaler"]["mean"])
                scaler.scale_ = np.array(saved_artifacts["scaler"]["scale"])
                scaler.var_ = scaler.scale_ ** 2 # approximate for sklearn
                X = scaler.transform(X)
                artifacts["scaler"] = saved_artifacts["scaler"]
            else:
                X = scaler.fit_transform(X)
                artifacts["scaler"] = {
                    "type": "StandardScaler",
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist()
                }
            
            full_dataset = DigitsDataset(X, y)
            total_size = len(full_dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size], generator=generator
            )
            
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
                
            # Handle missing values
            if cfg.missing_values == "drop":
                df = df.dropna()
            elif cfg.missing_values == "mean":
                df = df.fillna(df.mean(numeric_only=True))
                
            y = df[cfg.target_column].values
            X_df = df.drop(columns=[cfg.target_column])
            
            # Simple conversion, assumes numeric for now in this version
            try:
                X = X_df.values.astype(np.float32)
            except ValueError as e:
                raise ValueError("Could not convert CSV features to float32. Categorical encoding not fully supported yet.") from e
                
            # Split manually to fit scaler ONLY on train
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=config.seed)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=config.seed) # ~15% of total
            
            scaler = StandardScaler()
            if "scaler" in saved_artifacts:
                scaler.mean_ = np.array(saved_artifacts["scaler"]["mean"])
                scaler.scale_ = np.array(saved_artifacts["scaler"]["scale"])
                scaler.var_ = scaler.scale_ ** 2
                
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
