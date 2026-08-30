"""Structured event logger for training runs."""

from __future__ import annotations

import logging

from .base import BaseStore


class RunLogger:
    """Structured event logger for a training run."""

    def __init__(self, store: BaseStore, run_id: str, console: bool = True):
        self.store = store
        self.run_id = run_id
        self._logger = logging.getLogger(f"maskills.{run_id}")
        self._logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if not self._logger.handlers:
            log_path = store.get_log_path(run_id)
            if log_path:
                fh = logging.FileHandler(log_path)
                fh.setFormatter(logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
                self._logger.addHandler(fh)

            if console:
                ch = logging.StreamHandler()
                ch.setFormatter(logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
                self._logger.addHandler(ch)

    def info(self, msg: str):
        self._logger.info(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str, exc: Exception = None):
        self._logger.error(msg, exc_info=exc)

    def iteration_start(self, iteration: int, policies: dict[str, str]):
        self._logger.info(f"{'='*60}")
        self._logger.info(f"Iteration {iteration} started | {len(policies)} agents")

    def iteration_end(self, iteration: int, stats: dict):
        self._logger.info(
            f"Iteration {iteration} done | "
            f"avg_reward={stats.get('avg_reward', 0):.3f} | "
            f"tokens={stats.get('total_tokens', 0)} | "
            f"cost=${stats.get('cost_usd', 0):.4f}"
        )
        self.store.append_metrics(self.run_id, {"type": "iteration", "iteration": iteration, **stats})

    def episode_saved(self, iteration: int, episode_id: int, reward: float):
        self._logger.debug(f"  Episode {episode_id} saved | reward={reward:.3f}")

    def evaluation_done(self, iteration: int, episode_id, paradigm: str, raw_response: str):
        self._logger.debug(f"  Eval episode {episode_id} [{paradigm}]")
        self.store.save_evaluation(self.run_id, iteration, episode_id, {
            "episode_id": episode_id,
            "paradigm": paradigm,
            "raw_response": raw_response,
        })

    def gradient_saved(self, iteration: int, agent_id: str, num_gradients: int,
                       synthesis_method: str = "rewrite", momentum: float = 0.0,
                       momentum_applied: bool = False):
        diff_edit_tag = " | diff_edit" if synthesis_method == "diff_edit" else " | rewrite"
        if momentum > 0.0:
            mom_state = "applied" if momentum_applied else "skipped"
            mom_tag = f" | momentum=β{momentum:.2f} ({mom_state})"
        else:
            mom_tag = ""
        self._logger.info(
            f"  Gradient {agent_id} | {num_gradients} episode gradients aggregated"
            f"{diff_edit_tag}{mom_tag}"
        )
