import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader
from zenml import step

from src.config import DataConfig, TrainConfig
from src.data.dataset import DatasetBundle
from src.evaluators import Evaluator
from src.models.resnet18 import AnimalClassifierResNet18

LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluateModelOutput:
    accuracy: float
    precision: float
    recall: float
    f1: float


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def evaluate_model(
    model_path: str,
    data_bundle: DatasetBundle,
    data_config: DataConfig,
    train_config: TrainConfig,
) -> EvaluateModelOutput:
    """
    Evaluate the model on the test data.

    This step:
    - Receives the same data bundle from prepare_data_step (ensures same test set)
    - Loads the trained model from the provided path
    - Creates a DataLoader for the test dataset
    - Generates predictions for all test samples
    - Calculates evaluation metrics (accuracy, precision, recall, f1)
    - Logs metrics to MLflow

    Args:
        model_path (str): Path to the saved model checkpoint
        data_bundle (DatasetBundle): Bundle containing the test dataset (from prepare step)
        data_config (DataConfig): Configuration for data loading
        train_config (TrainConfig): Configuration for model initialization

    Returns:
        EvaluateModelOutput: Evaluation metrics containing accuracy, precision, recall, and f1 score
    """
    evaluator = Evaluator()

    LOGGER.info("Using test dataset from data bundle...")
    num_classes = len(data_bundle.label_names)

    LOGGER.info("Loading model from %s...", model_path)
    model = AnimalClassifierResNet18(
        num_classes=num_classes,
        optimizer=train_config.optimizer,
        pretrained=False,  # We're loading trained weights
        lr=train_config.lr,
        epochs=train_config.epochs,
        device=train_config.device,
        train_loader=None,  # Not needed for evaluation
        class_weights=None,
    )
    model.load(Path(model_path))

    LOGGER.info("Creating test data loader...")
    test_loader = DataLoader(
        data_bundle.test,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )

    LOGGER.info("Generating predictions on test data...")
    try:
        # Get predictions and true labels
        all_predictions = []
        all_labels = []

        model.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(model.device)
                outputs = model.model(images)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        y_pred = np.array(all_predictions)
        y_true = np.array(all_labels)

        LOGGER.info("Calculating evaluation metrics...")
        # Calculate the metrics
        accuracy, macro_precision, macro_recall, macro_f1 = (
            evaluator.compute_classification_metrics(
                y_pred=y_pred,
                y_true=y_true,
            )
        )

        LOGGER.info(
            "Test metrics - Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f",
            accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
        )

        mlflow.log_metrics(
            {
                "test_accuracy": accuracy,
                "test_precision": macro_precision,
                "test_recall": macro_recall,
                "test_f1": macro_f1,
            }
        )

        LOGGER.info("Evaluation completed successfully.")
        return EvaluateModelOutput(
            accuracy=accuracy,
            precision=macro_precision,
            recall=macro_recall,
            f1=macro_f1,
        )
    except Exception as e:
        LOGGER.error(
            "Exception %s occurred in evaluate_model step: %s",
            type(e).__name__,
            str(e),
        )
        raise e
