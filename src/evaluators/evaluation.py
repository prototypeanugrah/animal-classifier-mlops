import logging
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LOGGER = logging.getLogger(__name__)


@dataclass
class EvaluateModelOutput:
    accuracy: float
    precision: float
    recall: float
    f1: float


class Evaluator:
    def __init__(self) -> None:
        pass

    def compute_classification_metrics(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> EvaluateModelOutput:
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        (
            macro_precision,
            macro_recall,
            macro_f1,
            _,
        ) = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average="macro",
            zero_division=0,
        )

        LOGGER.info("Accuracy: %.4f.", accuracy)
        LOGGER.info("Macro Precision: %.4f.", macro_precision)
        LOGGER.info("Macro Recall: %.4f.", macro_recall)
        LOGGER.info("Macro F1: %.4f.", macro_f1)

        return EvaluateModelOutput(
            accuracy=accuracy,
            precision=macro_precision,
            recall=macro_recall,
            f1=macro_f1,
        )
