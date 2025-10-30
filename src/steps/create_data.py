import logging

from zenml import step

from src.config import DataConfig
from src.data.dataset import DatasetBundle, prepare_data_for_training
from src.materializers import DatasetBundleMaterializer

LOGGER = logging.getLogger(__name__)


@step(
    enable_cache=False,
    output_materializers=DatasetBundleMaterializer,
)
def prepare_data_step(data_config: DataConfig) -> DatasetBundle:
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
    data_bundle = prepare_data_for_training(data_config)

    LOGGER.info(
        "Data preparation complete. Train: %d, Val: %d, Test: %d samples",
        len(data_bundle.train),
        len(data_bundle.validation),
        len(data_bundle.test),
    )

    return data_bundle
