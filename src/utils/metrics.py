
class Metrics(NamedTuple):
    accuracy: float
    sensitivity: float
    specificity: float
    iou: float
    loss: float

def compute_metrics(
    y_pred: Float[Array, "N C H W"],
    y_true: Int[Array, "N H W"],
    loss: float
) -> Metrics:
    accuracy = compute_accuracy(y_pred, y_true)
    sensitivity = compute_sensitivity(y_pred, y_true)
    specificity = compute_specificity(y_pred, y_true)
    iou = compute_iou(y_pred, y_true)
    return Metrics( accuracy=accuracy, sensitivity=sensitivity, specificity=specificity, iou=iou, loss=loss)

def compute_accuracy(y_pred: Float[Array, "N C H W"], y_true: Int[Array, "N H W"]) -> float:
    y_pred = jnp.argmax(y_pred, axis=1)
    return jnp.mean(y_pred == y_true)

def compute_sensitivity(y_pred: Float[Array, "N C H W"], y_true: Int[Array, "N H W"]) -> float:
    y_pred = jnp.argmax(y_pred, axis=1)
    true_positives = jnp.sum((y_pred == 1) & (y_true == 1))
    false_negatives = jnp.sum((y_pred == 0) & (y_true == 1))
    return true_positives / (true_positives + false_negatives)

def compute_specificity(y_pred: Float[Array, "N C H W"], y_true: Int[Array, "N H W"]) -> float:
    y_pred = jnp.argmax(y_pred, axis=1)
    true_negatives = jnp.sum((y_pred == 0) & (y_true == 0))
    false_positives = jnp.sum((y_pred == 1) & (y_true == 0))
    return true_negatives / (true_negatives + false_positives)

def compute_iou(y_pred: Float[Array, "N C H W"], y_true: Int[Array, "N H W"]) -> float:
    y_pred = jnp.argmax(y_pred, axis=1)
    intersection = jnp.sum((y_pred == 1) & (y_true == 1))
    union = jnp.sum((y_pred == 1) | (y_true == 1))
    return intersection / union





