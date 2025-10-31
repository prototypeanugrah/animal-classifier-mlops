import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.pipelines.inference_pipeline import inference_pipeline

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the inference pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser


def run_inference(config: Path) -> None:
    """
    Run the inference pipeline using the provided configuration.

    Args:
        config (Path): Configuration file path.
    """
    LOGGER.info("Loading configuration from %s", config)
    pipeline_config = load_config(config)

    LOGGER.info("Starting inference pipeline...")
    pipeline_run = inference_pipeline(
        data_config=pipeline_config.data,
        inference_config=pipeline_config.inference,
    )

    run_id = getattr(pipeline_run, "id", None)
    if run_id:
        LOGGER.info("Inference pipeline completed. Run ID: %s", run_id)
    else:
        LOGGER.info("Inference pipeline completed.")
    LOGGER.info(
        "Inspect the 'predictor' step artifacts/logs for prediction details."
    )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_inference(args.config)
