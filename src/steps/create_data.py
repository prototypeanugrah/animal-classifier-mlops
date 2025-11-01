import logging
from pathlib import Path

import mlflow
import yaml
from zenml import step

from src.config import DataConfig
from src.data.dataset import DatasetBundle, prepare_data_for_training
from src.materializers import DatasetBundleMaterializer

LOGGER = logging.getLogger(__name__)


def _find_config_file(filename: str) -> Path:
    """Find the config.yaml file in common locations."""
    # Try current directory first (for temp files from deploy script)
    if Path(filename).exists():
        return Path(filename)

    # Try project root
    project_root = Path(__file__).parent.parent.parent
    if (project_root / filename).exists():
        return project_root / filename

    # Try src/steps/config.yaml
    if (project_root / "src" / "steps" / filename).exists():
        return project_root / "src" / "steps" / filename

    # Default fallback
    return Path(filename)


@step(
    # enable_cache=False,
    experiment_tracker="mlflow_tracker",
    output_materializers=DatasetBundleMaterializer,
)
def prepare_data_step() -> DatasetBundle:
    """
    Prepare data for training by loading, cleaning, and splitting into train/val/test sets.

    This step:
    - Loads the CSV data
    - Cleans and filters the data
    - Creates stratified train/validation/test splits
    - Returns a DatasetBundle with all three datasets

    Args:
        data_config (DataConfig): Configuration for data preparation

    Returns:
        DatasetBundle: Bundle containing train, validation, and test datasets
    """

    LOGGER.info("Preparing data for training...")
    config_path = _find_config_file("test_config.yaml")
    LOGGER.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        data_config = DataConfig(**config["data"])

    mlflow.log_params(data_config.model_dump(mode="json"))

    data_bundle = prepare_data_for_training(data_config)

    LOGGER.info(
        "Data preparation complete. Train: %d, Val: %d, Test: %d samples",
        len(data_bundle.train),
        len(data_bundle.validation),
        len(data_bundle.test),
    )

    return data_bundle
