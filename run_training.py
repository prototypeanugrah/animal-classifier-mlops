import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.pipelines.train_pipeline import train_pipeline

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the training pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser


def run_training(config: Path):
    """
    Run the training pipeline with the given configuration.

    Args:
        config (Path): Path to the configuration YAML file
    """
    LOGGER.info("Loading configuration from %s", config)
    pipeline_config = load_config(config)

    LOGGER.info("Starting training pipeline...")
    pipeline_run = train_pipeline(
        data_config=pipeline_config.data,
        train_config=pipeline_config.train,
    )

    run_id = getattr(pipeline_run, "id", None)
    if run_id:
        LOGGER.info("Training pipeline completed successfully! Run ID: %s", run_id)
    else:
        LOGGER.info("Training pipeline completed successfully!")
    LOGGER.info(
        "Evaluation metrics have been logged to MLflow. View them in the MLflow UI."
    )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_training(args.config)
