import logging

from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from src.steps.create_data import prepare_data_step
from src.steps.deploy_model import deployment_trigger
from src.steps.evaluate_model import evaluate_model
from src.steps.train_model import train_model

LOGGER = logging.getLogger(__name__)


@pipeline(enable_cache=False)
def deployment_pipeline() -> None:
    """
    Complete deployment pipeline for the animal classifier.

    This pipeline orchestrates the following steps:
    1. Data preparation: Load and split data once
    2. Model training: Train model using prepared data
    3. Model evaluation: Evaluate model on test data from same split
    4. Deploy model if evaluation metrics are good enough
    """

    data_bundle = prepare_data_step()
    model = train_model(data_bundle)
    metrics = evaluate_model(model, data_bundle)

    # Step 4: Deploy model if evaluation metrics are good enough
    deployment_decision = deployment_trigger(
        metrics=metrics,
    )

    model_deployment_service = mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
    )

    LOGGER.info("Model deployed successfully: %s", model_deployment_service)

    return model_deployment_service
