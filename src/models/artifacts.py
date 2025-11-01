from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainedModelArtifact:
    """Container capturing both local and MLflow references to a trained model."""

    model: torch.nn.Module
    local_path: Path
    mlflow_model_uri: str
