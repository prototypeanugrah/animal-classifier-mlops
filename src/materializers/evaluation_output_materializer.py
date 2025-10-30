from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Type

from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from src.evaluators import EvaluateModelOutput


class EvaluateModelOutputMaterializer(BaseMaterializer):
    """Materializer that stores evaluation metrics as JSON."""

    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA
    ASSOCIATED_TYPES = (EvaluateModelOutput,)

    _FILENAME = "metrics.json"

    def __init__(self, uri: str, artifact_store=None):
        super().__init__(uri, artifact_store)
        self._data_path = os.path.join(self.uri, self._FILENAME)

    def load(self, data_type: Type[EvaluateModelOutput]) -> EvaluateModelOutput:
        """Load evaluation metrics from the artifact store."""
        with self.artifact_store.open(self._data_path, "r") as handle:
            payload = json.load(handle)
        return EvaluateModelOutput(**payload)

    def save(self, data: EvaluateModelOutput) -> None:
        """Persist evaluation metrics as JSON."""
        if not fileio.exists(self.uri):
            fileio.makedirs(self.uri)
        with self.artifact_store.open(self._data_path, "w") as handle:
            json.dump(asdict(data), handle)
