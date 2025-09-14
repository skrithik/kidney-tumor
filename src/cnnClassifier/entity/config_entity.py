from dataclasses import dataclass #Automatically generates init, repr, eq, etc., for classes
from pathlib import Path

#entity
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path #Root folder for data ingestion artifacts.
    source_URL: str #URL to download dataset.
    local_data_file: Path #Where downloaded zip file is saved.
    unzip_dir: Path #Where extracted data will be stored.

#entity
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path  #Paths to save models.
    updated_base_model_path: Path
    params_image_size: list  #Model hyperparameters like image size, learning rate, number of output classes, etc.
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

#entity
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path  #Paths for data and model.
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int  #Training hyperparameters (epochs, batch size, augmentation flag).
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list
    

@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path  #Path to trained model.
    training_data: Path  #Evaluation data.
    all_params: dict
    mlflow_uri: str  #MLflow tracking URI.
    params_image_size: list
    params_batch_size: int  #Image size and batch size for evaluation.