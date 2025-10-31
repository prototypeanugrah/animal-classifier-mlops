from typing import Tuple

from src.config import DataConfig, TrainConfig
from src.evaluators import EvaluateModelOutput
from src.models import TrainedModelArtifact
from src.steps.create_data import prepare_data_step
from src.steps.evaluate_model import evaluate_model
from src.steps.train_model import train_model


def run_training_flow(
    data_config: DataConfig,
    train_config: TrainConfig,
) -> Tuple[TrainedModelArtifact, EvaluateModelOutput]:
    """
    Execute the shared training/evaluation flow.
    Used by multiple pipelines.

    Args:
        data_config (DataConfig): Configuration for data preparation and loading
        train_config (TrainConfig): Configuration for model training

    Returns:
        Tuple[TrainedModelArtifact, EvaluateModelOutput]: Trained model and evaluation metrics

    Note:
        Data is prepared once and passed to both training and evaluation steps,
        ensuring consistency. Evaluation metrics are logged to MLflow and can be
        viewed in the MLflow UI.
    """
    data_bundle = prepare_data_step(data_config)
    trained_model = train_model(data_bundle, data_config, train_config)
    metrics = evaluate_model(
        trained_model,
        data_bundle,
        data_config,
        train_config,
    )
    return trained_model, metrics
