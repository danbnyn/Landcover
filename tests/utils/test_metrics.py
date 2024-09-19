# test_metrics.py

import unittest
import jax.numpy as jnp
from typing import List, Any, Dict
import jax

from src.utils.metrics import Metric, get_metrics, ConfusionMatrixMetric, AccuracyMetric, IoUMetric, SensitivityMetric, SpecificityMetric

class TestMetricClasses(unittest.TestCase):
    def test_metric_abstract_class(self):
        """
        Test that Metric cannot be instantiated directly and that
        subclasses must implement all abstract methods.
        """
        with self.assertRaises(TypeError):
            metric = Metric()
    
    def test_get_metrics(self):
        """
        Test the get_metrics factory function.
        """
        metric_names = ["AccuracyMetric", "IoUMetric", "SensitivityMetric", "SpecificityMetric"]
        num_classes = 3
        metrics = get_metrics(metric_names, num_classes)
        self.assertEqual(len(metrics), 4)
        self.assertIsInstance(metrics[0], AccuracyMetric)
        self.assertIsInstance(metrics[1], IoUMetric)
        self.assertIsInstance(metrics[2], SensitivityMetric)
        self.assertIsInstance(metrics[3], SpecificityMetric)
        
        # Test with an invalid metric name
        with self.assertRaises(ValueError):
            get_metrics(["InvalidMetric"], num_classes)
    
    def test_confusion_matrix_metric_update(self):
        """
        Test the update method of ConfusionMatrixMetric.
        """
        num_classes = 2
        metric = ConfusionMatrixMetric(num_classes)
        
        # Create dummy data
        # Ground truth labels
        y_true_cls = jnp.array([
            [0, 0],
            [1, 1]
        ]).reshape(1, 2, 2)  # Shape: (N, H, W)
        
        # Predicted labels
        y_pred_cls = jnp.array([
            [0, 1],
            [1, 1]
        ]).reshape(1, 2, 2)  # Shape: (N, H, W)
        
        # Convert to one-hot encoding and reshape to (N, C, H, W)
        num_classes = 2
        y_true = jax.nn.one_hot(y_true_cls, num_classes=num_classes).transpose(0, 3, 1, 2)  # (N, C, H, W)
        y_pred = jax.nn.one_hot(y_pred_cls, num_classes=num_classes).transpose(0, 3, 1, 2)  # (N, C, H, W)
        
        metric.update(y_pred, y_true)
        
        # Expected confusion matrix components
        # For class 0:
        # TP = 1, FP = 0, FN = 1, TN = 2
        # For class 1:
        # TP = 2, FP = 1, FN = 0, TN = 1
        expected_tp = jnp.array([1, 2])
        expected_fp = jnp.array([0, 1])
        expected_fn = jnp.array([1, 0])
        expected_tn = jnp.array([2, 1])
        
        self.assertTrue(jnp.array_equal(metric.tp, expected_tp))
        self.assertTrue(jnp.array_equal(metric.fp, expected_fp))
        self.assertTrue(jnp.array_equal(metric.fn, expected_fn))
        self.assertTrue(jnp.array_equal(metric.tn, expected_tn))
    
    def test_accuracy_metric(self):
        """
        Test the AccuracyMetric computation.
        """
        num_classes = 2
        metric = AccuracyMetric(num_classes)
        
        # Create dummy data
        y_true_cls = jnp.array([
            [0, 1],
            [1, 0]
        ]).reshape(1, 2, 2)  # (N, H, W)
        
        y_pred_cls = jnp.array([
            [0, 1],
            [0, 0]
        ]).reshape(1, 2, 2)  # (N, H, W)
        
        y_true = jax.nn.one_hot(y_true_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        y_pred = jax.nn.one_hot(y_pred_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        
        metric.update(y_pred, y_true)
        results = metric.compute()
        
        # Manually compute expected per-class and global accuracies
        # For class 0:
        # TP = 2 (pixels where class 0 was correctly predicted)
        # FP = 1 (pixels where class 0 was predicted but was actually class 1)
        # FN = 0 (pixels where class 0 was not predicted but was actually class 0)
        # TN = 1 (pixels where class 0 was not predicted and was not class 0)
        # Accuracy = (TP + TN) / (TP + FP + TN + FN)
        acc_class_0 = (2 + 1) / (2 + 1 + 1 + 0) 
        
        # For class 1:
        # TP = 1, FP = 0, FN = 1, TN = 2
        acc_class_1 = (1 + 2) / (1 + 0 + 2 + 1) 
        
        expected_per_class_accuracy = [0.75, 0.75]
        expected_global_accuracy = (3) / (4) 
        
        self.assertEqual(results['per_class_accuracy'], expected_per_class_accuracy)
        self.assertAlmostEqual(results['global_accuracy'], expected_global_accuracy, places=7)
    
    def test_iou_metric(self):
        """
        Test the IoUMetric computation.
        """
        num_classes = 2
        metric = IoUMetric(num_classes)
        
        # Create dummy data
        y_true_cls = jnp.array([
            [0, 1],
            [1, 0]
        ]).reshape(1, 2, 2)
        
        y_pred_cls = jnp.array([
            [0, 0],
            [1, 0]
        ]).reshape(1, 2, 2)
        
        y_true = jax.nn.one_hot(y_true_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        y_pred = jax.nn.one_hot(y_pred_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        
        metric.update(y_pred, y_true)
        results = metric.compute()
        
        # Manually compute IoU for each class
        # For class 0:
        # TP = 2, FP = 1, FN = 0
        iou_class_0 = 2 / (2 + 1 + 0) 
        # For class 1:
        # TP = 0, FP = 0, FN = 2
        iou_class_1 = 0 / (0 + 0 + 2) 
        
        expected_per_class_iou = [2/3, 0.0]
        expected_mean_iou = (2/3 + 0) / 2 
        
        self.assertAlmostEqual(results['per_class_iou'][0], expected_per_class_iou[0], places=4)
        self.assertAlmostEqual(results['per_class_iou'][1], expected_per_class_iou[1], places=4)
        self.assertAlmostEqual(results['mean_iou'], expected_mean_iou, places=4)
    
    def test_sensitivity_metric(self):
        """
        Test the SensitivityMetric computation.
        """
        num_classes = 2
        metric = SensitivityMetric(num_classes)
        
        # Create dummy data
        y_true_cls = jnp.array([
            [0, 1],
            [1, 0]
        ]).reshape(1, 2, 2)
        
        y_pred_cls = jnp.array([
            [0, 0],
            [1, 0]
        ]).reshape(1, 2, 2)
        
        y_true = jax.nn.one_hot(y_true_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        y_pred = jax.nn.one_hot(y_pred_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        
        metric.update(y_pred, y_true)
        results = metric.compute()
        
        # Sensitivity (Recall) = TP / (TP + FN)
        # For class 0:
        # TP = 2, FN = 0
        sensitivity_class_0 = 2 / (2 + 0)
        # For class 1:
        # TP = 0, FN = 2
        sensitivity_class_1 = 0 / (0 + 2) 
        
        expected_per_class_sensitivity = [1.0, 0.0]
        expected_mean_sensitivity = (1.0 + 0.0) / 2
        
        self.assertEqual(results['per_class_sensitivity'], expected_per_class_sensitivity)
        self.assertAlmostEqual(results['mean_sensitivity'], expected_mean_sensitivity, places=7)
    
    def test_specificity_metric(self):
        """
        Test the SpecificityMetric computation.
        """
        num_classes = 2
        metric = SpecificityMetric(num_classes)
        
        # Create dummy data
        y_true_cls = jnp.array([
            [0, 1],
            [1, 0]
        ]).reshape(1, 2, 2)
        
        y_pred_cls = jnp.array([
            [0, 0],
            [1, 0]
        ]).reshape(1, 2, 2)
        
        y_true = jax.nn.one_hot(y_true_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        y_pred = jax.nn.one_hot(y_pred_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        
        metric.update(y_pred, y_true)
        results = metric.compute()
        
        # Specificity = TN / (TN + FP)
        # For class 0:
        # TN = 0, FP = 1
        specificity_class_0 = 0 / (0 + 1 + 1e-7) 
        # For class 1:
        # TN = 2, FP = 0
        specificity_class_1 = 2 / (2 + 0)
        
        expected_per_class_specificity = [0.0, 1.0]
        expected_mean_specificity = (0.0 + 1.0) / 2 
        
        self.assertEqual(results['per_class_specificity'], expected_per_class_specificity)
        self.assertAlmostEqual(results['mean_specificity'], expected_mean_specificity, places=7)
    
    def test_reset_method(self):
        """
        Test the reset method of ConfusionMatrixMetric and its subclasses.
        """
        num_classes = 2
        metric = IoUMetric(num_classes)
        
        # Create dummy data
        y_true_cls = jnp.array([[0]])
        y_pred_cls = jnp.array([[0]])
        
        y_true = jax.nn.one_hot(y_true_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        y_pred = jax.nn.one_hot(y_pred_cls, num_classes=num_classes).transpose(0, 3, 1, 2)
        
        metric.update(y_pred, y_true)
        metric.reset()
        
        self.assertTrue(jnp.array_equal(metric.tp, jnp.zeros(num_classes)))
        self.assertTrue(jnp.array_equal(metric.fp, jnp.zeros(num_classes)))
        self.assertTrue(jnp.array_equal(metric.fn, jnp.zeros(num_classes)))
        self.assertTrue(jnp.array_equal(metric.tn, jnp.zeros(num_classes)))

if __name__ == '__main__':
    unittest.main()
