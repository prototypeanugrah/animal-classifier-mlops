from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainedModelArtifact:
    """Container capturing both local and MLflow references to a trained model."""

    local_path: Path
    mlflow_model_uri: str
