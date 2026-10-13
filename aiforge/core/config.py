from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Optional, List, Dict, Any
import yaml
import os

class EarlyStoppingConfig(BaseModel):
    enabled: bool = Field(False, description="Enable early stopping.")
    metric: str = Field("val_loss", description="Metric to monitor (e.g., 'val_loss', 'val_accuracy').")
    mode: str = Field("min", description="Mode: 'min' or 'max'.")
    patience: int = Field(5, description="Number of epochs with no improvement after which training will be stopped.")
    min_delta: float = Field(0.0001, description="Minimum change in the monitored quantity to qualify as an improvement.")
    min_epochs: int = Field(5, description="Minimum number of epochs to train before early stopping can trigger.")

class SchedulerConfig(BaseModel):
    enabled: bool = Field(False, description="Enable learning rate scheduler.")
    type: str = Field("reduce_lr_on_plateau", description="Type of scheduler.")
    factor: float = Field(0.5, description="Factor by which the learning rate will be reduced.")
    patience: int = Field(3, description="Number of epochs with no improvement after which learning rate will be reduced.")
    min_lr: float = Field(1e-6, description="A lower bound on the learning rate.")

class CSVDatasetConfig(BaseModel):
    path: str = Field(..., description="Path to the CSV file.")
    target_column: str = Field(..., description="The column to predict.")
    exclude_columns: List[str] = Field(default_factory=list, description="Columns to ignore.")
    missing_values: str = Field("drop", description="Strategy for missing values: 'drop' or 'mean'.")

class ImageDatasetConfig(BaseModel):
    path: str = Field(..., description="Path to the images root directory.")
    image_size: List[int] = Field([64, 64], description="Target [height, width] for images.")
    mode: str = Field("folders", description="Loading mode: 'folders' (expects train/val/test subdirs) or 'single' (random split).")

class DataConfig(BaseModel):
    type: str = Field(..., description="Type of data: 'digits_builtin', 'csv', 'images'.")
    csv_config: Optional[CSVDatasetConfig] = None
    image_config: Optional[ImageDatasetConfig] = None

class MLPConfig(BaseModel):
    input_size: int = Field(..., description="Size of the input layer.")
    hidden_sizes: List[int] = Field([64, 32], description="Sizes of hidden layers.")
    output_size: int = Field(..., description="Size of the output layer (number of classes).")

class CNNConfig(BaseModel):
    in_channels: int = Field(3, description="Number of input channels (e.g. 3 for RGB).")
    conv_layers: List[int] = Field([32, 64], description="Number of filters for each conv layer.")
    hidden_sizes: List[int] = Field([128], description="Dense layers after convolutions.")
    output_size: int = Field(..., description="Size of the output layer (number of classes).")

class LLMConfig(BaseModel):
    model_id: str = Field(..., description="HF Model ID")
    allow_download: bool = Field(False)
    max_seq_length: int = Field(128)
    lora_enabled: bool = Field(False)
    lora_r: int = Field(8)
    lora_alpha: int = Field(16)
    lora_dropout: float = Field(0.05)

class ArchitectureConfig(BaseModel):
    type: str = Field(..., description="Type of architecture: 'mlp', 'cnn', 'llm'.")
    mlp_config: Optional[MLPConfig] = None
    cnn_config: Optional[CNNConfig] = None
    llm_config: Optional[LLMConfig] = None

class DistillationConfig(BaseModel):
    enabled: bool = Field(False, description="Enable knowledge distillation")
    teacher_experiment_id: str = Field("", description="Registry ID of the teacher experiment")
    temperature: float = Field(2.0, description="Temperature for distillation", gt=0.0)
    alpha: float = Field(0.5, description="Weight for distillation loss", ge=0.0, le=1.0)

class ParameterSearchSpace(BaseModel):
    name: str = Field(..., description="Parameter name (e.g., 'learning_rate', 'distillation.temperature')")
    type: str = Field(..., description="'float', 'int', or 'categorical'")
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[list] = None
    log: bool = False

class TuningConfig(BaseModel):
    enabled: bool = Field(False, description="Enable hyperparameter tuning")
    trials: int = Field(10, description="Number of tuning trials")
    metric: str = Field("val_loss", description="Metric to optimize")
    mode: str = Field("min", description="'min' or 'max'")
    search_space: List[ParameterSearchSpace] = Field(default_factory=list)
    study_name: Optional[str] = None
    storage: Optional[str] = None

class ExperimentConfig(BaseModel):
    experiment_name: str = Field("Experiment", description="The name of the experiment.")
    domain: str = Field("vision", description="The domain of the experiment, e.g., 'vision', 'tabular'.")
    task: str = Field("classification", description="The specific task, e.g., 'classification'.")
    
    data: DataConfig
    architecture: ArchitectureConfig
    
    batch_size: int = Field(32, description="Batch size for training and evaluation.")
    epochs: int = Field(5, description="Number of epochs for training.")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer.")
    device: str = Field("auto", description="Device to use for training: 'auto', 'cpu', 'mps', or 'cuda'.")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility.")
    backend: str = Field("native", description="Training backend to use: 'native' or 'soup'.")
    backend_args: Dict[str, Any] = Field(default_factory=dict, description="Additional arguments for the backend.")
    
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)

    @model_validator(mode='after')
    def validate_compatibility(self) -> 'ExperimentConfig':
        if self.architecture.type == "cnn" and self.data.type not in ["images"]:
            raise ValueError("CNN architecture is only compatible with 'images' data type.")
        if self.architecture.type == "mlp" and self.data.type == "images":
            # For simplicity in v0.4, we restrict MLP to tabular/digits
            raise ValueError("MLP architecture expects tabular/flattened data in this version.")
        
        if self.data.type == "csv" and not self.data.csv_config:
            raise ValueError("csv_config is required when data type is 'csv'.")
        if self.data.type == "images" and not self.data.image_config:
            raise ValueError("image_config is required when data type is 'images'.")
            
        if self.architecture.type == "mlp" and not self.architecture.mlp_config:
            raise ValueError("mlp_config is required when architecture is 'mlp'.")
        if self.architecture.type == "cnn" and not self.architecture.cnn_config:
            raise ValueError("cnn_config is required when architecture is 'cnn'.")
            
        return self

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")
            
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format in {yaml_path}. Expected a dictionary.")
            
        # Migrate old v0.2/v0.3 configs on the fly
        if "data" not in data:
            # It's an old config format
            dataset_name = data.get("dataset_name", "digits")
            data["data"] = {"type": f"{dataset_name}_builtin"}
            if "dataset_name" in data: del data["dataset_name"]
            
        if "architecture" not in data:
            data["architecture"] = {
                "type": "mlp",
                "mlp_config": {
                    "input_size": data.pop("input_size", 64),
                    "hidden_sizes": data.pop("hidden_sizes", [128, 64]),
                    "output_size": data.pop("output_size", 10)
                }
            }
            
        try:
            return cls(**data)
        except ValidationError as e:
            print(f"Error parsing configuration {yaml_path}:")
            for err in e.errors():
                loc = ".".join(map(str, err['loc']))
                print(f" - {loc}: {err['msg']}")
            raise ValueError("Configuration validation failed.") from e
