def compute_metrics(
    y_pred: Float[Array, "N C H W"],
    y_true: Int[Array, "N H W"],
    loss: float
) -> Metrics:
    accuracy = jnp.mean(jnp.argmax(y_pred, axis=1) == y_true)
    iou = compute_iou(y_pred, y_true)  # Implement this function
    return Metrics(loss, accuracy, iou)

