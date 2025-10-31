import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from zenml.steps import BaseParameters, step
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService

from src.config import DataConfig, InferenceConfig
from src.data.dataloader import build_transforms

LOGGER = logging.getLogger(__name__)


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """Parameters used to locate a deployed MLflow prediction service."""

    pipeline_name: str
    step_name: str
    running: bool = True


@step(enable_cache=False)
def load_image_for_inference(
    data_config: DataConfig,
    inference_config: InferenceConfig,
) -> np.ndarray:
    """Load a single image from disk and convert it into a numpy array."""
    image_path = inference_config.input_image_path
    if image_path is None:
        raise ValueError(
            "Set inference.input_image_path in the configuration before running inference."
        )

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Inference image not found at {path}.")

    _, eval_transform = build_transforms(image_size=data_config.image_size)
    image = Image.open(path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0)
    LOGGER.info("Loaded inference image from %s", path)
    return tensor.numpy()


@step(enable_cache=False)
def prediction_service_loader(
    params: MLFlowDeploymentLoaderStepParameters,
    inference_config: InferenceConfig,
) -> Optional[MLFlowDeploymentService]:
    """Retrieve the MLflow prediction service deployed by the pipeline."""
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = model_deployer.find_model_server(
        pipeline_name=params.pipeline_name,
        pipeline_step_name=params.step_name,
        running=params.running,
    )

    if not services:
        LOGGER.warning(
            "No MLflow prediction service deployed by step '%s' in pipeline '%s'.",
            params.step_name,
            params.pipeline_name,
        )
        return None

    service = services[0]
    try:
        service.start(timeout=inference_config.service_wait_seconds)
    except Exception:  # pragma: no cover - already running or cannot start
        LOGGER.debug("Prediction service already running.")

    try:
        if hasattr(service, "wait_for_service_ready"):
            service.wait_for_service_ready(timeout=inference_config.service_wait_seconds)
        else:
            service.wait(timeout_seconds=inference_config.service_wait_seconds)
    except Exception as exc:  # pragma: no cover - external service failure
        LOGGER.warning("Prediction service did not report ready state: %s", exc)

    return service


@step(enable_cache=False)
def predictor(
    service: Optional[MLFlowDeploymentService],
    image_array: np.ndarray,
) -> np.ndarray:
    """Run an inference request via the MLflow service."""
    if service is None:
        raise RuntimeError(
            "No running MLflow service is available. Run the deployment pipeline first."
        )

    LOGGER.info(
        "Requesting prediction from MLflow service at %s.",
        service.prediction_url,
    )
    prediction = service.predict(image_array)
    return np.asarray(prediction)
