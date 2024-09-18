from jaxtyping import Array, Float, Int
from typing import NamedTuple
import jax.numpy as jnp
import tensorflow as tf
from typing import List
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


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

    def reset(self):
        """
        Reset the metric state for a new computation.
        """
        self.__init__()


def get_metrics(metric_names: List[str], num_classes: int) -> List[Metric]:
    """
    Factory function to create metric instances based on their names.

    Args:
        metric_names (List[str]): List of metric class names as strings.
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
    def __init__(self, num_classes: int):
        """
        Initializes the confusion matrix components.
        
        Args:
            num_classes (int): Number of classes.
        """
        self.num_classes = num_classes
        self.tp = jnp.zeros(num_classes)
        self.fp = jnp.zeros(num_classes)
        self.tn = jnp.zeros(num_classes)
        self.fn = jnp.zeros(num_classes)

    def reset(self):
        self.tp = jnp.zeros(self.num_classes, dtype=jnp.float32)
        self.fp = jnp.zeros(self.num_classes, dtype=jnp.float32)
        self.tn = jnp.zeros(self.num_classes, dtype=jnp.float32)
        self.fn = jnp.zeros(self.num_classes, dtype=jnp.float32)

    
    def update(self, y_pred: jnp.ndarray, y_true: jnp.ndarray):
        """
        Updates the confusion matrix components.
        
        Args:
            y_pred (jnp.ndarray): One-hot encoded predictions (N, C, H, W).
            y_true (jnp.ndarray): One-hot encoded ground truth labels (N, C, H, W).
        """
        # Flatten the arrays to (N*H*W, C)
        y_pred_flat = y_pred.reshape(-1, self.num_classes)
        y_true_flat = y_true.reshape(-1, self.num_classes)
        
        # Compute TP, FP, TN, FN per class
        tp = jnp.sum(y_pred_flat * y_true_flat, axis=0)
        fp = jnp.sum(y_pred_flat * (1 - y_true_flat), axis=0)
        fn = jnp.sum((1 - y_pred_flat) * y_true_flat, axis=0)
        tn = jnp.sum((1 - y_pred_flat) * (1 - y_true_flat), axis=0)
        
        # Accumulate
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
    
    def compute(self):
        """
        Computes confusion matrix components.
        
        Returns:
            Tuple containing TP, FP, TN, FN arrays.
        """
        return self.tp, self.fp, self.tn, self.fn


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

def log_metrics(metrics: Dict[str, Any], writer: tf.summary.SummaryWriter, step: int, class_names: Optional[List[str]] = None):
    """
    Logs metrics to TensorBoard dynamically without specifying each metric.

    Args:
        metrics (Dict[str, Any]): Dictionary containing all metric values.
        writer (tf.summary.SummaryWriter): TensorBoard writer.
        step (int): Current training step or epoch.
        class_names (Optional[List[str]], optional): List of class names for labeling. Defaults to None.
    """
    with writer.as_default():
        for metric_name, metric_value in metrics.items():
            if "per_class" in metric_name:
                # Extract the type of metric, e.g., 'accuracy' from 'per_class_accuracy'
                metric_type = metric_name.replace("per_class_", "").capitalize()
                for idx, val in enumerate(metric_value):
                    class_label = f"Class_{idx}" if class_names is None else class_names[idx]
                    writer_name = f"{metric_type}/Class_{idx}" if class_names is None else f"{metric_type}/{class_label}"
                    tf.summary.scalar(writer_name, float(val), step=step)
            elif "mean_" in metric_name:
                # Extract the metric type, e.g., 'iou' from 'mean_iou'
                metric_type = metric_name.replace("mean_", "").capitalize()
                writer_name = f"Mean {metric_type}"
                tf.summary.scalar(writer_name, float(metric_value), step=step)
            else:
                # Handle other potential global metrics
                writer_name = metric_name.replace("_", " ").capitalize()
                tf.summary.scalar(writer_name, float(metric_value), step=step)