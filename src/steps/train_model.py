import logging
from pathlib import Path

import mlflow
import torch
from mlflow.models import infer_signature
from zenml import step

from src.config import DataConfig, TrainConfig
from src.data.dataset import DatasetBundle, build_data_loaders
from src.materializers import TrainedModelArtifactMaterializer
from src.models import TrainedModelArtifact
from src.models.resnet18 import AnimalClassifierResNet18

LOGGER = logging.getLogger(__name__)


@step(
    enable_cache=False,
    experiment_tracker="mlflow_tracker",
    output_materializers=TrainedModelArtifactMaterializer,
)
def train_model(
    data_bundle: DatasetBundle,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> TrainedModelArtifact:
    """
    Train the model on the prepared data.

    This step:
    - Receives prepared data from the prepare_data_step
    - Creates DataLoaders from the DatasetBundle
    - Initializes the model
    - Trains the model with train and validation data
    - Saves the best model checkpoint
    - Logs the best model to MLflow for downstream deployment

    Args:
        data_bundle (DatasetBundle): Bundle containing train, validation, and test datasets
        data_config (DataConfig): Configuration for data loading
        train_config (TrainConfig): Configuration for training

    Returns:
        TrainedModelArtifact: References to the trained model artifact
    """
    LOGGER.info("Building data loaders...")
    loaders = build_data_loaders(
        data_bundle,
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
    )

    LOGGER.info("Initializing model with %d classes...", len(data_bundle.label_names))
    model = AnimalClassifierResNet18(
        num_classes=len(data_bundle.label_names),
        optimizer=train_config.optimizer,
        pretrained=train_config.pretrained,
        lr=train_config.lr,
        max_lr=train_config.max_lr,
        epochs=train_config.epochs,
        device=train_config.device,
        train_loader=loaders["train"],
        class_weights=None,
    )

    LOGGER.info("Starting model training...")
    model.fit(
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        save_dir=train_config.save_dir,
        save_best_only=train_config.save_best_only,
    )

    model_path = Path(train_config.save_dir) / "best_model.pth"
    LOGGER.info(
        "Loading best checkpoint from %s for MLflow logging",
        model_path,
    )
    model.load(model_path)

    example_input = torch.randn(
        1,
        3,
        data_config.image_size,
        data_config.image_size,
        device="cpu",
    )
    model.model.eval()
    with torch.no_grad():
        example_output = model.model(example_input.to(model.device)).cpu().numpy()

    example_input_numpy = example_input.cpu().numpy()
    signature = infer_signature(
        example_input_numpy,
        example_output,
    )

    artifact_path = train_config.mlflow_model_name
    LOGGER.info(
        "Logging trained model to MLflow at artifact path '%s'",
        artifact_path,
    )
    mlflow.pytorch.log_model(
        pytorch_model=model.model,
        artifact_path=artifact_path,
        input_example=example_input_numpy,
        signature=signature,
        registered_model_name=train_config.mlflow_model_name,
    )
    mlflow_model_uri = mlflow.get_artifact_uri(artifact_path)
    LOGGER.info("Model logged to MLflow with URI %s", mlflow_model_uri)

    LOGGER.info("Model training completed.")
    return TrainedModelArtifact(
        local_path=model_path,
        mlflow_model_uri=mlflow_model_uri,
    )
