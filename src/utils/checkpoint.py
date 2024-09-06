
import equinox as eqx
import jax
import optax
import json
import os
import heapq
from typing import Any, NamedTuple
import pathlib

class Checkpoint(NamedTuple):
    model: Any
    model_state: Any
    optimizer_state: optax.OptState
    step: int
    metric_value: float

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.checkpoints = []

    def save_checkpoint(self, checkpoint: Checkpoint):
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint.step}.eqx"
        eqx.tree_serialise_leaves(checkpoint_path, checkpoint)

        self.checkpoints.append((checkpoint.metric_value, checkpoint_path))
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)

        if len(self.checkpoints) > self.max_to_keep:
            _, path_to_remove = self.checkpoints.pop()
            path_to_remove.unlink()

    def load_best_checkpoint(self):
        if not self.checkpoints:
            return None

        _, best_checkpoint_path = self.checkpoints[0]
        return self.load_checkpoint(best_checkpoint_path)

    @staticmethod
    def load_checkpoint(checkpoint_path):
        dummy_checkpoint = Checkpoint(None, None, None, 0, 0.0)
        loaded_checkpoint = eqx.tree_deserialise_leaves(checkpoint_path, dummy_checkpoint)
        return loaded_checkpoint

