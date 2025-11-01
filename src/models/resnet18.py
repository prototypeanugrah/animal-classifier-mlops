"""ResNet18 model for image classification."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

LOGGER = logging.getLogger(__name__)


class AnimalClassifierResNet18(torch.nn.Module):
    """ResNet18-based animal classifier with training pipeline."""

    def __init__(
        self,
        num_classes: int,
        optimizer: str,
        pretrained: bool,
        lr: float,
        max_lr: float,
        epochs: int,
        device: str,
        train_loader: Optional[DataLoader],
        class_weights: Optional[torch.Tensor],
    ):
        """
        Initialize the ResNet18 classifier.

        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
            lr: Learning rate for optimizer
            epochs: Number of training epochs
            device: Device to use for training ('cuda', 'mps' or 'cpu')
            train_loader: Training data loader (required for scheduler setup)
            class_weights: Optional class weights for loss function
        """
        super().__init__()
        self._num_classes = num_classes
        self._lr = lr
        self._max_lr = max_lr
        self._epochs = epochs

        requested_device = device.lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available!")
            if torch.backends.mps.is_available():
                LOGGER.info("MPS is available, using MPS instead of CPU.")
                requested_device = "mps"
            else:
                LOGGER.info("No CUDA or MPS available, using CPU.")
                requested_device = "cpu"
        self.device = requested_device

        # Initialize model
        self.model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, self._num_classes)
        self.model = self.model.to(self.device)

        # Initialize optimizer and loss
        _OPTIMIZERS = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
        }
        self._optimizer = _OPTIMIZERS[optimizer](self.model.parameters(), lr=lr)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self._criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Initialize scheduler if train_loader is provided
        self._scheduler = None
        if train_loader is not None:
            self._scheduler = OneCycleLR(
                self._optimizer,
                max_lr=max_lr,
                epochs=epochs,
                steps_per_epoch=len(train_loader),
            )

        # Metrics tracking
        self._best_val_loss = float("inf")
        self._best_val_acc = 0.0
        self._training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self._optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self._criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self._optimizer.step()

            # Update scheduler if available
            if self._scheduler is not None:
                self._scheduler.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if batch_idx % 10 == 0:
                LOGGER.info(
                    "Batch [%s/%s] - Loss: %s - Acc: %s%%",
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    100.0 * correct / total,
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self._criterion(outputs, labels)

                # Track metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: Optional[Path] = None,
        save_best_only: bool = True,
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save model checkpoints
            save_best_only: Whether to save only the best model

        Returns:
            Dictionary containing training history
        """
        LOGGER.info("Starting training on %s", self.device)
        LOGGER.info("Number of epochs: %s", self._epochs)
        LOGGER.info("Learning rate: %s", self._lr)
        LOGGER.info("Maximum learning rate: %s", self._max_lr)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self._epochs):
            LOGGER.info("\nEpoch %s/%s", epoch + 1, self._epochs)
            LOGGER.info("-" * 50)

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            LOGGER.info(
                "Training - Loss: %s - Accuracy: %s%%",
                train_loss,
                train_acc,
            )

            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)
            LOGGER.info(
                "Validation - Loss: %s - Accuracy: %s%%",
                val_loss,
                val_acc,
            )

            # Update history
            self._training_history["train_loss"].append(train_loss)
            self._training_history["train_acc"].append(train_acc)
            self._training_history["val_loss"].append(val_loss)
            self._training_history["val_acc"].append(val_acc)

            # Log metrics to MLflow for real-time monitoring
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            mlflow.log_metric(
                "learning_rate",
                self._scheduler.get_last_lr()[0],
                step=epoch,
            )

            # Save model checkpoint
            if save_dir is not None:
                if save_best_only:
                    if val_acc > self._best_val_acc:
                        self._best_val_acc = val_acc
                        self._best_val_loss = val_loss
                        best_model_path = save_dir / "best_model.pth"
                        self.save(best_model_path)
                        LOGGER.info(
                            "Saved best model with validation accuracy: %s%%",
                            val_acc,
                        )
                else:
                    checkpoint_path = save_dir / f"model_epoch_{epoch + 1}.pth"
                    self.save(checkpoint_path)
                    LOGGER.info("Saved checkpoint: %s", checkpoint_path)

        LOGGER.info("\nTraining completed!")
        LOGGER.info("Best validation accuracy: %s%%", self._best_val_acc)
        LOGGER.info("Best validation loss: %s", self._best_val_loss)

        return self._training_history

    def save(self, path: Path) -> None:
        """
        Save model state dict.

        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)
        LOGGER.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        """
        Load model state dict.

        Args:
            path: Path to load the model from
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model = self.model.to(self.device)
        LOGGER.info("Model loaded from %s", path)

    def get_training_history(self) -> Dict[str, list]:
        """Get the training history."""
        return self._training_history

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Generate predictions for a data loader.

        Args:
            data_loader: DataLoader to generate predictions for

        Returns:
            torch.Tensor: Predicted class indices
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                predictions.append(preds.cpu())

        return torch.cat(predictions)
