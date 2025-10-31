import logging

from zenml.steps import step

from src.config import EvaluationConfig
from src.evaluators import EvaluateModelOutput

LOGGER = logging.getLogger(__name__)


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
    precision_threshold_met = metrics.precision >= config.min_precision
    recall_threshold_met = metrics.recall >= config.min_recall
    f1_threshold_met = metrics.f1 >= config.min_f1
    accuracy_threshold_met = metrics.accuracy >= config.min_accuracy

    LOGGER.info(
        "Deployment gate results - precision: %.4f (>= %.4f: %s), "
        "recall: %.4f (>= %.4f: %s), f1: %.4f (>= %.4f: %s), "
        "accuracy: %.4f (>= %.4f: %s)",
        metrics.precision,
        config.min_precision,
        precision_threshold_met,
        metrics.recall,
        config.min_recall,
        recall_threshold_met,
        metrics.f1,
        config.min_f1,
        f1_threshold_met,
        metrics.accuracy,
        config.min_accuracy,
        accuracy_threshold_met,
    )

    conditions_met = (
        precision_threshold_met
        and recall_threshold_met
        and f1_threshold_met
        and accuracy_threshold_met
    )

    LOGGER.info("Deployment decision: %s", "approved" if conditions_met else "rejected")
    return conditions_met
