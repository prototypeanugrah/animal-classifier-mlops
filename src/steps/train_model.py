import logging
from pathlib import Path

from zenml import step

from src.config import DataConfig, TrainConfig
from src.data.dataset import DatasetBundle, build_data_loaders
from src.models.resnet18 import AnimalClassifierResNet18

LOGGER = logging.getLogger(__name__)


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model(
    data_bundle: DatasetBundle,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> str:
    """
    Train the model on the prepared data.

    This step:
    - Receives prepared data from the prepare_data_step
    - Creates DataLoaders from the DatasetBundle
    - Initializes the model
    - Trains the model with train and validation data
    - Saves the best model checkpoint
    - Returns the path to the saved model

    Args:
        data_bundle (DatasetBundle): Bundle containing train, validation, and test datasets
        data_config (DataConfig): Configuration for data loading
        train_config (TrainConfig): Configuration for training

    Returns:
        str: Path to the saved model checkpoint
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
    LOGGER.info("Model training completed. Model saved to %s", model_path)
    return str(model_path)
