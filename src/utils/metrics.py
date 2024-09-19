from jaxtyping import Array, Float, Int
from typing import NamedTuple
import jax.numpy as jnp
import tensorflow as tf
from typing import List
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from tensorboardX import SummaryWriter
import numpy as np

class Metric(ABC):
    """
    Abstract base class for all metrics.
    """
    @abstractmethod
    def __init__(self):
        """
        Initialize any necessary variables.
        """
        pass

    @abstractmethod
    def update(self, y_pred: Any, y_true: Any):
        """
        Update the metric state with new predictions and labels.
        
        Args:
            y_pred: Predicted values (e.g., one-hot encoded predictions).
            y_true: Ground truth labels (e.g., one-hot encoded labels).
        """
        pass

    @abstractmethod
    def compute(self):
        """
        Compute the final metric value after all updates.
        
        Returns:
            The computed metric value.
        """
        pass


def get_metrics(metric_names: List[str], num_classes: int) -> List[Metric]:
    """
    Factory function to create metric instances based on their names.

    Args:
        metric_names (List[str]): List of metric class names.
        num_classes (int): Number of classes.

    Returns:
        List[Metric]: List of metric instances.
    """
    metric_map = {
        "AccuracyMetric": AccuracyMetric,
        "IoUMetric": IoUMetric,
        "SensitivityMetric": SensitivityMetric,
        "SpecificityMetric": SpecificityMetric
    }

    metrics = []
    for name in metric_names:
        metric_class = metric_map.get(name)
        if metric_class is None:
            raise ValueError(f"Metric '{name}' is not recognized.")
        metrics.append(metric_class(num_classes=num_classes))
    
    return metrics

class ConfusionMatrixMetric(Metric):
    """
    Base class for metrics that rely on the confusion matrix.
    Accumulates a confusion matrix and per-class TP, FP, TN, FN.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        # Initialize confusion matrix: rows = true classes, columns = predicted classes
        self.confusion_matrix = jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
        # Initialize TP, FP, TN, FN per class
        self.tp = jnp.zeros(num_classes, dtype=jnp.int32)
        self.fp = jnp.zeros(num_classes, dtype=jnp.int32)
        self.fn = jnp.zeros(num_classes, dtype=jnp.int32)
        self.tn = jnp.zeros(num_classes, dtype=jnp.int32)

    def reset(self):
        self.confusion_matrix = jnp.zeros((self.num_classes, self.num_classes), dtype=jnp.int32)
        self.tp = jnp.zeros(self.num_classes, dtype=jnp.int32)
        self.fp = jnp.zeros(self.num_classes, dtype=jnp.int32)
        self.fn = jnp.zeros(self.num_classes, dtype=jnp.int32)
        self.tn = jnp.zeros(self.num_classes, dtype=jnp.int32)

    def update(self, y_pred_cls: jnp.ndarray, y_true_cls: jnp.ndarray):
        """
        Update confusion matrix and TP, FP, TN, FN per class in a vectorized manner.

        Args:
            y_pred (jnp.ndarray): Predictions, shape (N, H, W).
            y_true (jnp.ndarray): Ground truth labels, shape (N, H, W).
        """

        # Compute confusion matrix
        indices = y_true_cls * self.num_classes + y_pred_cls
        counts = jnp.bincount(indices, minlength=self.num_classes**2)
        cm = counts.reshape((self.num_classes, self.num_classes))
        self.confusion_matrix += cm

        # Update TP, FP, FN, TN per class
        self.tp += jnp.diag(cm)
        self.fp += jnp.sum(cm, axis=0) - jnp.diag(cm)
        self.fn += jnp.sum(cm, axis=1) - jnp.diag(cm)
        self.tn += self.num_classes * self.num_classes - (self.fp + self.fn + self.tp)

    def compute_confusion_matrix(self) -> jnp.ndarray:
        """
        Returns the accumulated confusion matrix.

        Returns:
            jnp.ndarray: Confusion matrix, shape (C, C).
        """
        return self.confusion_matrix
    
    def compute(self):

        pass

class AccuracyMetric(ConfusionMatrixMetric):
    """
    Computes both per-class accuracy and global accuracy.
    """
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def compute(self) -> Dict[str, Any]:
        """
        Compute per-class and global accuracy.

        Returns:
            Dict[str, Any]: Dictionary containing per-class and global accuracy.
        """
        per_class_accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn + 1e-7)
        global_accuracy = jnp.sum(self.tp + self.tn) / (jnp.sum(self.tp + self.fp + self.tn + self.fn) + 1e-7)


        return {
            "per_class_accuracy": per_class_accuracy.tolist(),
            "global_accuracy": float(global_accuracy)
        }


class IoUMetric(ConfusionMatrixMetric):
    """
    Computes both per-class IoU and mean IoU.
    """
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def compute(self) -> Dict[str, Any]:
        """
        Compute per-class IoU and mean IoU.

        Returns:
            Dict[str, Any]: Dictionary containing per-class IoU and mean IoU.
        """
        intersection = self.tp
        union = self.tp + self.fp + self.fn
        per_class_iou = intersection / (union + 1e-7)
        mean_iou = jnp.mean(per_class_iou)

        return {
            "per_class_iou": per_class_iou.tolist(),
            "mean_iou": float(mean_iou)
        }

class SensitivityMetric(ConfusionMatrixMetric):
    """
    Computes both per-class sensitivity (recall) and mean sensitivity.
    """
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def compute(self) -> Dict[str, Any]:
        """
        Compute per-class sensitivity and mean sensitivity.

        Returns:
            Dict[str, Any]: Dictionary containing per-class sensitivity and mean sensitivity.
        """
        sensitivity = self.tp / (self.tp + self.fn + 1e-7)
        mean_sensitivity = jnp.mean(sensitivity)

        return {
            "per_class_sensitivity": sensitivity.tolist(),
            "mean_sensitivity": float(mean_sensitivity)
        }

class SpecificityMetric(ConfusionMatrixMetric):
    """
    Computes both per-class specificity and mean specificity.
    """
    def __init__(self, num_classes: int):
        super().__init__(num_classes)

    def compute(self) -> Dict[str, Any]:
        """
        Compute per-class specificity and mean specificity.

        Returns:
            Dict[str, Any]: Dictionary containing per-class specificity and mean specificity.
        """
        specificity = self.tn / (self.tn + self.fp + 1e-7)
        mean_specificity = jnp.mean(specificity)

        return {
            "per_class_specificity": specificity.tolist(),
            "mean_specificity": float(mean_specificity)
        }


def log_metrics(metrics: Dict[str, Any], writer: SummaryWriter, step: int, class_names: Optional[List[str]] = None):
    """
    Logs metrics to TensorBoard dynamically without specifying each metric.
    
    Args:
        metrics (Dict[str, Any]): Dictionary containing all metric values.
        writer (SummaryWriter): TensorBoardX writer.
        step (int): Current training step or epoch.
        class_names (List[str], optional): List of class names for labeling. Defaults to None.
    """
    for metric_name, metric_value in metrics.items():
        if "per_class" in metric_name:
            # Extract the type of metric, e.g., 'accuracy' from 'per_class_accuracy'
            metric_type = metric_name.replace("per_class_", "").capitalize()
            scalars_dict = {}
            for idx, val in enumerate(metric_value):
                class_label = f"Class_{idx}" if class_names is None else class_names[idx]
                scalars_dict[class_label] = float(val)
            writer.add_scalars(f"{metric_type}/Per_Class", scalars_dict, global_step=step)
        elif "mean_" in metric_name:
            # Extract the metric type, e.g., 'iou' from 'mean_iou'
            metric_type = metric_name.replace("mean_", "").capitalize()
            writer.add_scalar(f"Mean_{metric_type}", float(metric_value), global_step=step)
        else:
            # Handle other potential global metrics
            writer.add_scalar(metric_name.replace("_", " ").capitalize(), float(metric_value), global_step=step)