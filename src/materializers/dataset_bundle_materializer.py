from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Type

import torchvision.transforms as T
from zenml.enums import ArtifactType
from zenml.io import fileio
from zenml.materializers.base_materializer import BaseMaterializer

from src.data.dataset import AnimalDataset, DatasetBundle, ImageRecord
from src.data.dataloader import build_transforms


class DatasetBundleMaterializer(BaseMaterializer):
    """Custom materializer that stores DatasetBundle objects as JSON."""

    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA
    ASSOCIATED_TYPES = (DatasetBundle,)

    _FILENAME = "dataset_bundle.json"

    def __init__(self, uri: str, artifact_store=None):
        super().__init__(uri, artifact_store)
        self._data_path = os.path.join(self.uri, self._FILENAME)

    def load(self, data_type: Type[DatasetBundle]) -> DatasetBundle:
        """Load a DatasetBundle from the artifact store."""
        with self.artifact_store.open(self._data_path, "r") as handle:
            payload = json.load(handle)

        label_names = payload["label_names"]
        label_ids = payload["label_ids"]

        train_dataset = self._deserialize_dataset(payload["train"], label_names)
        validation_dataset = self._deserialize_dataset(
            payload["validation"], label_names
        )
        test_dataset = self._deserialize_dataset(payload["test"], label_names)

        return DatasetBundle(
            train=train_dataset,
            validation=validation_dataset,
            test=test_dataset,
            label_names=label_names,
            label_ids=label_ids,
        )

    def save(self, data: DatasetBundle) -> None:
        """Persist a DatasetBundle as JSON metadata."""
        if not fileio.exists(self.uri):
            fileio.makedirs(self.uri)

        serialized = {
            "label_names": data.label_names,
            "label_ids": data.label_ids,
            "train": self._serialize_dataset(data.train),
            "validation": self._serialize_dataset(data.validation),
            "test": self._serialize_dataset(data.test),
        }

        with self.artifact_store.open(self._data_path, "w") as handle:
            json.dump(serialized, handle)

    def _serialize_dataset(self, dataset: AnimalDataset) -> Dict[str, Any]:
        transform_meta = self._extract_transform_metadata(dataset)
        cache_dir = str(dataset._cache_dir) if dataset._cache_dir else None
        records = [
            {
                "uuid": record.uuid,
                "image_url": record.image_url,
                "label_index": record.label_index,
            }
            for record in dataset._records
        ]

        return {
            "records": records,
            "cache_dir": cache_dir,
            "transform": transform_meta,
        }

    def _deserialize_dataset(
        self,
        serialized: Dict[str, Any],
        label_names: List[str],
    ) -> AnimalDataset:
        image_size = serialized["transform"]["image_size"]
        transform_variant = serialized["transform"]["variant"]

        train_transform, eval_transform = build_transforms(image_size)
        transform = train_transform if transform_variant == "train" else eval_transform

        cache_dir_value = serialized.get("cache_dir")
        cache_dir = Path(cache_dir_value) if cache_dir_value else None

        records = [
            ImageRecord(
                uuid=record["uuid"],
                image_url=record["image_url"],
                label_index=record["label_index"],
            )
            for record in serialized["records"]
        ]

        return AnimalDataset(
            records=records,
            label_names=label_names,
            transform=transform,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _extract_transform_metadata(dataset: AnimalDataset) -> Dict[str, Any]:
        transform = dataset._transform
        if not isinstance(transform, T.Compose):
            raise ValueError("Expected dataset transform to be torchvision Compose.")

        transforms_list = list(transform.transforms)

        variant = (
            "train"
            if any(isinstance(t, T.RandomHorizontalFlip) for t in transforms_list)
            else "eval"
        )

        image_size = None
        for t in transforms_list:
            if isinstance(t, T.Resize):
                size = t.size
                image_size = size[0] if isinstance(size, (tuple, list)) else size
                break

        if image_size is None:
            raise ValueError("Unable to determine image size from dataset transform.")

        return {
            "variant": variant,
            "image_size": image_size,
        }
