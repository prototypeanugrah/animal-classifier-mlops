import argparse
import logging
from pathlib import Path

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from src.config import load_config
from src.pipelines.deployment_pipeline import deployment_pipeline

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deployment pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser


def run_deployment(config: Path):
    """
    Run the deployment pipeline with the given configuration.

    Args:
        config (Path): Path to the configuration YAML file
    """
    LOGGER.info("Loading configuration from %s", config)
    pipeline_config = load_config(config)

    LOGGER.info("Starting deployment pipeline...")
    pipeline_run = deployment_pipeline(
        data_config=pipeline_config.data,
        train_config=pipeline_config.train,
        evaluation_config=pipeline_config.evaluation,
    )

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    service = model_deployer.find_model_server(
        pipeline_name="deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=True,
    )

    if service:
        first_service = service[0]
        if first_service:
            print(
                "The MLflow prediction server is running locally as a daemon process "
                "and accepts inference requests at:\n"
                f"    {first_service.prediction_url}\n"
            )

    run_id = getattr(pipeline_run, "id", None)
    if run_id:
        LOGGER.info("Deployment pipeline completed successfully! Run ID: %s", run_id)
    else:
        LOGGER.info("Deployment pipeline completed successfully!")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_deployment(args.config)
