from jaxtyping import Float, Int, Array, PyTree
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable

# Generalized loss function creator
def create_loss_fn(loss_type="weighted_bce_loss", **kwargs):
    if loss_type == "weighted_bce_loss":
        def loss_fn(predictions, targets, weights):
            return weighted_bce_loss(predictions, targets, weights)
    else:
        raise ValueError(f"Unknown loss function type: {loss_type}")

    return loss_fn

def weighted_bce_loss(predictions, targets, weights=None):
    """
    Compute weighted binary cross-entropy loss.

    Args:
        predictions: jnp.ndarray of shape (batch, num_classes, h, w), predicted probabilities.
        targets: jnp.ndarray of shape (batch, num_classes, h, w), ground truth labels (0 or 1).
        weights: jnp.ndarray of shape (num_classes,), weights to apply to each class.

    Returns:
        loss: the weighted binary cross-entropy loss.
    """

    # Compute binary cross-entropy loss for each element
    bce_loss = - (targets * jnp.clip(jnp.log(predictions), min = -100) + (1 - targets) * jnp.clip(jnp.log(1 - predictions), min = -100))

    # Apply weights if provided
    if weights is not None:
        # Reshape weights to match the shape of the bce_loss (batch, num_classes, h, w)
        weights = weights.reshape((1, -1, 1, 1))  # Reshape to (1, num_classes, 1, 1)
        bce_loss = bce_loss * weights

    return jnp.mean(bce_loss)

def batch_loss_fn(
    model: eqx.Module,
    state: eqx.nn.State,
    x_true: Float[Array, " N C H W"],
    y_true: Int[Array, " N C H W"],
    weights: Float[Array, "C"],
    loss_fn: Callable[[Float[Array, "N C H W"], Int[Array, "N C H W"], Float[Array, "C"]], Float[Array, "..."]],
) -> PyTree[Float[Array, "..."]]:

    batch_model = jax.vmap(
        model, axis_name='batch', in_axes=(0,None), out_axes=(0,None)
    )
    y_pred, new_state = batch_model(x_true, state)
    loss = loss_fn(y_pred, y_true, weights)

    return loss, new_state
