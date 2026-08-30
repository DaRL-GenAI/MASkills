"""Verified reward computation for LOCOMO QA episodes.

Mirrors ``env/locomo/task_eval/evaluation.py`` so the training signal
matches what the upstream benchmark scores.  Category routing:

* Category 1 (multi-hop): comma-split partial F1 averaged over sub-answers.
* Categories 2/3/4 (temporal, single-hop, open-domain): stemmed token F1.
  For category 3 only the first ``;``-segment of the gold answer is used,
  matching ``eval_question_answering`` in the upstream file.
* Category 5 (adversarial): 1.0 iff the response contains
  ``"not mentioned"`` or ``"no information available"``; else 0.0.

The score is in ``[0.0, 1.0]`` and serves directly as the Monte-Carlo
reward consumed by the centralized critic / optimizer.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Tuple

from maskills.core.base import BaseReward, Trajectory

from .metrics import compute_locomo_metrics

try:
    from nltk.stem import PorterStemmer  # type: ignore
    _STEMMER = PorterStemmer()

    def _stem(token: str) -> str:
        return _STEMMER.stem(token)
except ImportError:  # graceful fallback when NLTK is unavailable
    def _stem(token: str) -> str:
        return token


_PUNCT = set(string.punctuation)
_ARTICLE_RE = re.compile(r"\b(a|an|the|and)\b")
_WS_RE = re.compile(r"\s+")
_CAT5_POSITIVES = ("no information available", "not mentioned")


def _normalize_answer(text: str) -> str:
    """Replica of upstream ``normalize_answer``."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFD", str(text))
    text = text.replace(",", "")
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLE_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _f1_token(prediction: str, ground_truth: str) -> float:
    pred_tokens = [_stem(w) for w in _normalize_answer(prediction).split()]
    gt_tokens = [_stem(w) for w in _normalize_answer(ground_truth).split()]
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _multi_answer_f1(prediction: str, ground_truth: str) -> float:
    preds = [p.strip() for p in prediction.split(",") if p.strip()]
    gts = [g.strip() for g in ground_truth.split(",") if g.strip()]
    if not preds or not gts:
        return 0.0
    per_gt = []
    for gt in gts:
        per_gt.append(max(_f1_token(p, gt) for p in preds))
    return sum(per_gt) / len(per_gt)


def _cat5_score(prediction: str) -> float:
    low = (prediction or "").lower()
    return 1.0 if any(p in low for p in _CAT5_POSITIVES) else 0.0


def score_locomo_qa(prediction: str, ground_truth: str, category: int) -> float:
    """Return [0,1] reward matching upstream LOCOMO scoring."""
    if category == 5:
        return _cat5_score(prediction)
    if category == 1:
        return _multi_answer_f1(prediction, ground_truth)
    if category in (2, 3, 4):
        if category == 3:
            ground_truth = (ground_truth or "").split(";")[0].strip()
        return _f1_token(prediction, ground_truth)
    # Unknown category: be conservative, score with plain F1.
    return _f1_token(prediction, ground_truth)


class LocomoRewardGenerator(BaseReward):
    """Reward generator implementing the LOCOMO scoring rules."""

    def __init__(self, reward_metric: str = "f1_bleu_mean"):
        """
        Args:
            reward_metric: what ``compute`` returns as the training reward —
                ``"f1"`` (category-routed token F1), ``"bleu"`` (category-
                routed BLEU), or ``"f1_bleu_mean"`` (their mean, so both F1
                and BLEU drive the optimization — the default).
        """
        # Kept for symmetry with VerifiedRewardGenerator (which needs an
        # LLM client for QA judging); we don't require one here.
        self._client = None
        self.reward_metric = reward_metric

    def set_client(self, client):  # noqa: D401 - no-op, kept for API parity
        self._client = client

    def compute(self, trajectory: Trajectory) -> float:
        task = trajectory.task or {}
        category = int(task.get("category", 0) or 0)
        ground_truth = str(task.get("ground_truth", ""))
        final_answer = trajectory.metadata.get("final_answer", "")
        if not final_answer and trajectory.steps:
            last = trajectory.steps[-1]
            final_answer = last.get("output", last.get("action", ""))

        score = score_locomo_qa(final_answer or "", ground_truth, category)

        # Category-routed F1 + BLEU from the upstream LOCOMO evaluator,
        # stored on the trajectory for per-category metric reporting.
        metrics = {}
        try:
            metrics = compute_locomo_metrics(
                final_answer or "", ground_truth, category,
            )
            trajectory.metadata["qa_metrics"] = metrics
        except Exception:  # never fail the rollout on a metric error
            pass

        # The training reward: F1, BLEU, or their mean (so both drive the
        # optimization).  Falls back to the category-routed score.
        if self.reward_metric == "f1":
            return float(metrics.get("f1", score))
        if self.reward_metric == "bleu":
            return float(metrics.get("bleu", 0.0))
        if self.reward_metric == "f1_bleu_mean" and metrics:
            return float((metrics.get("f1", 0.0) + metrics.get("bleu", 0.0)) / 2.0)
        return float(score)

    # Exposed for diagnostics / callbacks if desired.
    @staticmethod
    def score(prediction: str, ground_truth: str, category: int) -> Tuple[float, str]:
        return score_locomo_qa(prediction, ground_truth, category), "ok"
