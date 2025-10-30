import logging
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
)

LOGGER = logging.getLogger(__name__)


class Evaluator:
    def __init__(self) -> None:
        pass

    def compute_classification_metrics(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Tuple[float, float, float, float]:
        try:
            accuracy = self.accuracy_score(y_pred=y_pred, y_true=y_true)
            macro_precision, macro_recall, macro_f1, _ = (
                precision_recall_fscore_support(
                    y_true=y_true,
                    y_pred=y_pred,
                    average="macro",
                    zero_division=0,
                )
            )

            LOGGER.info("Accuracy: %s.", accuracy)
            LOGGER.info("Macro Precision: %s.", macro_precision)
            LOGGER.info("Macro Recall: %s.", macro_recall)
            LOGGER.info("Macro F1: %s.", macro_f1)

            return accuracy, macro_precision, macro_recall, macro_f1
        except Exception as e:
            LOGGER.info(
                "Exception %s occurred in compute_classification_metrics method of the Evaluation class. Exception message:  ",
                str(e),
            )
            raise Exception()
