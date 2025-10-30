from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from src.config import DataConfig, EvaluationConfig, TrainConfig
from src.steps.create_data import prepare_data_step
from src.steps.evaluate_model import evaluate_model
from src.steps.train_model import train_model
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

    # Step 1: Prepare data once (creates train/val/test splits)
    data_bundle = prepare_data_step(data_config)

    # Step 2: Train model using prepared data
    model_path = train_model(data_bundle, data_config, train_config)

    # Step 3: Evaluate model using same test data from the bundle
    metrics = evaluate_model(
        model_path,
        data_bundle,
        data_config,
        train_config,
    )

    # Step 4: Deploy model if evaluation metrics are good enough
    deployment_decision = deployment_trigger(
        metrics=metrics,
        config=evaluation_config,
    )

    model_deployer = mlflow_model_deployer_step.with_options(
        parameters={
            "model_name": "animal-classifier-resnet18",
            "workers": 2,
        },
    )
    model_deployer(model=model_path, deploy_decision=deployment_decision)
