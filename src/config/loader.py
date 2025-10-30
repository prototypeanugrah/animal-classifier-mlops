from __future__ import annotations

from pathlib import Path

import yaml

from .schema import PipelineConfig


def load_config(config_path: Path) -> PipelineConfig:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return PipelineConfig.model_validate(config)
