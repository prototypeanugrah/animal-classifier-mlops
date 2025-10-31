from zenml import pipeline
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW

from src.config import DataConfig, TrainConfig
from src.pipelines._shared import run_training_flow

docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline(data_config: DataConfig, train_config: TrainConfig):
    """
    Complete training pipeline for the animal classifier.

    This pipeline orchestrates the following steps:
    1. Data preparation: Load and split data once
    2. Model training: Train model using prepared data
    3. Model evaluation: Evaluate model on test data from same split

    Args:
        data_config (DataConfig): Configuration for data preparation and loading
        train_config (TrainConfig): Configuration for model training

    Note:
        Data is prepared once and passed to both training and evaluation steps,
        ensuring consistency. Evaluation metrics are logged to MLflow and can be
        viewed in the MLflow UI.
    """

    run_training_flow(data_config, train_config)
