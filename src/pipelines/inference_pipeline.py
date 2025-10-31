from zenml import pipeline

from src.config import DataConfig, InferenceConfig
from src.steps.inference import (
    MLFlowDeploymentLoaderStepParameters,
    load_image_for_inference,
    prediction_service_loader,
    predictor,
)


@pipeline(enable_cache=False)
def inference_pipeline(
    data_config: DataConfig,
    inference_config: InferenceConfig,
):
    """Simple inference pipeline that calls the deployed MLflow service."""

    image_array = load_image_for_inference(
        data_config=data_config,
        inference_config=inference_config,
    )

    service = prediction_service_loader(
        params=MLFlowDeploymentLoaderStepParameters(
            pipeline_name=inference_config.pipeline_name,
            step_name=inference_config.pipeline_step_name,
            running=inference_config.running,
        ),
        inference_config=inference_config,
    )

    predictions = predictor(
        service=service,
        image_array=image_array,
    )

    return predictions
