"""Training callback system."""

from abc import ABC


class Callback(ABC):
    """Base callback class for training hooks."""

    def on_iteration_start(self, iteration: int, trainer): ...
    def on_iteration_end(self, iteration: int, stats: dict, trainer): ...
    def on_episode_complete(self, trajectory, trainer): ...
    def on_policy_update(self, agent_id: str, old_policy: str, new_policy: str): ...


class LoggingCallback(Callback):
    """Structured logging callback."""

    def on_iteration_start(self, iteration: int, trainer):
        trainer.run_logger.iteration_start(iteration, trainer.checkpoint.get_policies())

    def on_iteration_end(self, iteration: int, stats: dict, trainer):
        trainer.run_logger.iteration_end(iteration, stats)


class CheckpointCallback(Callback):
    """Save checkpoints after each iteration."""

    def on_iteration_end(self, iteration: int, stats: dict, trainer):
        # Checkpoint saving is handled in the trainer itself
        pass


class EarlyStoppingCallback(Callback):
    """Stop training on convergence."""

    def __init__(self, patience: int = 3, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self._best_reward = float('-inf')
        self._wait = 0

    def on_iteration_end(self, iteration: int, stats: dict, trainer):
        reward = stats.get('avg_reward', 0)
        if reward > self._best_reward + self.min_delta:
            self._best_reward = reward
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                trainer.run_logger.info(
                    f"Early stopping: no improvement for {self.patience} iterations"
                )
                trainer._should_stop = True
