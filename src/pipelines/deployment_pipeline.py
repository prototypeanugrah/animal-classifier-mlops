from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from src.config import DataConfig, EvaluationConfig, TrainConfig
from src.pipelines._shared import run_training_flow
from src.steps.deploy_model import deployment_trigger


@pipeline(enable_cache=False)
def deployment_pipeline(
    data_config: DataConfig,
    train_config: TrainConfig,
    evaluation_config: EvaluationConfig,
) -> None:
    """
    Complete deployment pipeline for the animal classifier.

    This pipeline orchestrates the following steps:
    1. Data preparation: Load and split data once
    2. Model training: Train model using prepared data
    3. Model evaluation: Evaluate model on test data from same split
    4. Deploy model if evaluation metrics are good enough

    Args:
        data_config (DataConfig): Configuration for data preparation and loading
        train_config (TrainConfig): Configuration for model training
        evaluation_config (EvaluationConfig): Configuration for evaluation
    """

    # Steps 1-3: Reuse the shared training and evaluation flow
    trained_model, metrics = run_training_flow(data_config, train_config)

    # Step 4: Deploy model if evaluation metrics are good enough
    deployment_decision = deployment_trigger(
        metrics=metrics,
        config=evaluation_config,
    )

    model_deployer = mlflow_model_deployer_step.with_options(
        parameters={
            "model_name": train_config.mlflow_model_name,
            "workers": 2,
        },
    )

    model_deployer(
        model=trained_model,
        deploy_decision=deployment_decision,
    )
