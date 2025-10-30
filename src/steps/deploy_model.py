from zenml.steps import step

from src.config import EvaluationConfig
from src.evaluators import EvaluateModelOutput


@step(enable_cache=False)
def deployment_trigger(
    metrics: EvaluateModelOutput,
    config: EvaluationConfig,
) -> bool:
    """Determine if the evaluated model is good enough to deploy.

    Args:
        metrics (EvaluateModelOutput): Aggregated evaluation metrics.
        config (EvaluationConfig): Thresholds to compare against.

    Returns:
        bool: True to deploy the model, False otherwise.
    """
    precision_threshold_met = metrics.precision > config.min_precision
    recall_threshold_met = metrics.recall > config.min_recall
    conditions_met = precision_threshold_met and recall_threshold_met
    return conditions_met
