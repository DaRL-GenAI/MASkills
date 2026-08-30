"""Standard QA metrics for the HotpotQA path of the language env.

* **F1 / precision / recall / EM** — re-exported from the upstream
  ``hotpot_evaluate_v1.py`` (``env/lang_benchmark/HotPotQA/hotpot``) so we
  score with the official tokenisation / normalisation used by the
  HotpotQA leaderboard.
* **BLEU** — corpus-style sentence BLEU via NLTK, with the same
  HotpotQA ``normalize_answer`` applied to both sides and method-1
  smoothing so the very short, often unigram answers do not collapse to
  zero.

Both metrics are computed alongside the LLM-as-judge reward in
``reward.py`` and stashed in trajectory metadata under ``qa_metrics``.
They are diagnostic — the training signal still comes from the judge.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict

# ── Upstream HotpotQA eval (official F1/EM) ──────────────────────────

def _load_upstream_eval():
    """Import ``hotpot_evaluate_v1`` dynamically from the env tree.

    The file ships with the benchmark and is not a Python package, so we
    load it by path rather than via ``import``.
    """
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "env" / "lang_benchmark" / "HotPotQA" / "hotpot" / "hotpot_evaluate_v1.py"
    if not path.exists():
        raise FileNotFoundError(
            f"hotpot_evaluate_v1.py not found at {path}. "
            "The HotpotQA benchmark directory is required."
        )
    spec = importlib.util.spec_from_file_location("_hotpot_eval_v1", path)
    module = importlib.util.module_from_spec(spec)
    # The upstream file imports ujson; fall back to stdlib json if missing.
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError:
        import json as _json
        import sys
        import types
        sys.modules.setdefault("ujson", types.SimpleNamespace(
            load=_json.load, loads=_json.loads, dump=_json.dump, dumps=_json.dumps,
        ))
        spec.loader.exec_module(module)
    return module


_upstream = None


def _get_upstream():
    global _upstream
    if _upstream is None:
        _upstream = _load_upstream_eval()
    return _upstream


def normalize_answer(text: str) -> str:
    """Upstream HotpotQA ``normalize_answer`` (lowercase, strip articles/punct)."""
    return _get_upstream().normalize_answer(text or "")


def f1_score(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Upstream HotpotQA token F1.  Returns ``{f1, precision, recall}``."""
    f1, prec, rec = _get_upstream().f1_score(prediction or "", ground_truth or "")
    return {"f1": float(f1), "precision": float(prec), "recall": float(rec)}


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(_get_upstream().exact_match_score(prediction or "", ground_truth or ""))


# ── BLEU (NLTK sentence_bleu w/ smoothing) ───────────────────────────

def bleu_score(prediction: str, ground_truth: str) -> float:
    """Sentence BLEU with HotpotQA normalisation and method-1 smoothing.

    HotpotQA answers are typically 1-5 tokens, so we use BLEU-1..BLEU-4
    with equal weights and smoothing to keep the score informative on
    short hypotheses.  Returns 0.0 when either side has no tokens.
    """
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return 0.0

    ref = normalize_answer(ground_truth).split()
    hyp = normalize_answer(prediction).split()
    if not ref or not hyp:
        return 0.0
    n = min(4, len(ref), len(hyp))
    weights = tuple(1.0 / n for _ in range(n))
    smoother = SmoothingFunction().method1
    try:
        return float(sentence_bleu([ref], hyp, weights=weights, smoothing_function=smoother))
    except (ZeroDivisionError, ValueError):
        return 0.0


# ── Aggregator used by the reward generator ──────────────────────────

def compute_qa_metrics(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Return a dict with F1, precision, recall, EM, and BLEU."""
    out = f1_score(prediction, ground_truth)
    out["em"] = exact_match(prediction, ground_truth)
    out["bleu"] = bleu_score(prediction, ground_truth)
    return out
