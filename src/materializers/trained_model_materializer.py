import json
from pathlib import Path
from typing import Type

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from src.models import TrainedModelArtifact


class TrainedModelArtifactMaterializer(BaseMaterializer):
    """Persist TrainedModelArtifact objects as JSON payloads."""

    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL
    ASSOCIATED_TYPES = (TrainedModelArtifact,)

    _FILENAME = "trained_model_artifact.json"

    def __init__(self, uri: str, artifact_store=None):
        super().__init__(uri, artifact_store)
        self._artifact_path = Path(self.uri) / self._FILENAME

    def load(self, data_type: Type[TrainedModelArtifact]) -> TrainedModelArtifact:
        with self.artifact_store.open(str(self._artifact_path), "r") as fp:
            payload = json.load(fp)

        return TrainedModelArtifact(
            local_path=Path(payload["local_path"]),
            mlflow_model_uri=payload["mlflow_model_uri"],
        )

    def save(self, artifact: TrainedModelArtifact) -> None:
        if not fileio.exists(self.uri):
            fileio.makedirs(self.uri)

        payload = {
            "local_path": str(artifact.local_path),
            "mlflow_model_uri": artifact.mlflow_model_uri,
        }

        with self.artifact_store.open(str(self._artifact_path), "w") as fp:
            json.dump(payload, fp)
