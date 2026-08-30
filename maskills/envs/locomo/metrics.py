"""Standard QA metrics for LOCOMO.

* **F1** — re-uses the upstream LOCOMO evaluator
  (``env/locomo/task_eval/evaluation.py``).  Routing matches
  ``eval_question_answering``:

    - cat 1 (multi-hop)   → ``evaluation.f1`` (comma-split multi-answer F1)
    - cat 2/3/4           → ``evaluation.f1_score`` (stemmed token F1; for
      cat 3, only the first ``;``-segment of the gold answer is used)
    - cat 5 (adversarial) → 1 if the response contains
      ``"not mentioned"`` / ``"no information available"`` else 0
      (this is what the upstream evaluator scores for cat 5)

* **BLEU** — NLTK ``sentence_bleu`` with method-1 smoothing, applied
  after LOCOMO's ``normalize_answer`` + Porter stemming so the token
  surface matches what F1 sees.  For multi-answer (cat 1) gold answers
  we max BLEU across the comma-split sub-answers, mirroring the F1
  routing.

Both metrics are diagnostic; the reward generator stores them in the
trajectory metadata next to the category-routed reward.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict

# ── Upstream LOCOMO evaluator (official F1) ──────────────────────────

def _load_upstream_eval():
    """Import ``env/locomo/task_eval/evaluation.py`` dynamically.

    The file lives outside the Python package tree, so we load it by path.
    It imports several heavy optional deps (``bert_score``, ``rouge``);
    those imports are at module top-level, so they will fail loudly if
    missing.  We pre-stub them so the F1/normalisation helpers we want
    can be loaded even on a minimal install.
    """
    import sys
    import types

    # Pre-stub heavy deps the upstream file pulls in but we don't need.
    if "bert_score" not in sys.modules:
        stub = types.ModuleType("bert_score")
        stub.score = lambda *a, **k: ([0.0], [0.0], [0.0])
        sys.modules["bert_score"] = stub
    if "rouge" not in sys.modules:
        stub = types.ModuleType("rouge")
        class _DummyRouge:  # noqa: D401 - tiny stub for rouge.Rouge
            def get_scores(self, *a, **k):
                return {"rouge-1": {"f": 0.0}}
        stub.Rouge = _DummyRouge
        sys.modules["rouge"] = stub

    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "env" / "locomo" / "task_eval" / "evaluation.py"
    if not path.exists():
        raise FileNotFoundError(
            f"LOCOMO evaluation.py not found at {path}. "
            "The LOCOMO benchmark directory is required."
        )
    spec = importlib.util.spec_from_file_location("_locomo_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_upstream = None


def _get_upstream():
    global _upstream
    if _upstream is None:
        _upstream = _load_upstream_eval()
    return _upstream


def normalize_answer(text: str) -> str:
    return _get_upstream().normalize_answer(text or "")


def f1_score(prediction: str, ground_truth: str, category: int) -> float:
    """Category-aware F1 mirroring upstream ``eval_question_answering``."""
    ev = _get_upstream()
    cat = int(category or 0)
    if cat == 5:
        low = (prediction or "").lower()
        return 1.0 if ("not mentioned" in low or "no information available" in low) else 0.0
    if cat == 1:
        return float(ev.f1(prediction or "", ground_truth or ""))
    if cat == 3:
        ground_truth = (ground_truth or "").split(";")[0].strip()
    return float(ev.f1_score(prediction or "", ground_truth or ""))


# ── BLEU (NLTK sentence_bleu, normalised to match F1) ────────────────

def _bleu_pair(pred_norm: str, gt_norm: str) -> float:
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        return 0.0
    ref = gt_norm.split()
    hyp = pred_norm.split()
    if not ref or not hyp:
        return 0.0
    n = min(4, len(ref), len(hyp))
    weights = tuple(1.0 / n for _ in range(n))
    smoother = SmoothingFunction().method1
    try:
        return float(sentence_bleu([ref], hyp, weights=weights, smoothing_function=smoother))
    except (ZeroDivisionError, ValueError):
        return 0.0


def _stemmed_norm(text: str) -> str:
    """Apply LOCOMO normalisation + Porter stemming (matches f1_score)."""
    ev = _get_upstream()
    try:
        from nltk.stem import PorterStemmer
        ps = PorterStemmer()
        return " ".join(ps.stem(w) for w in ev.normalize_answer(text or "").split())
    except ImportError:
        return ev.normalize_answer(text or "")


def bleu_score(prediction: str, ground_truth: str, category: int) -> float:
    """Category-aware BLEU mirroring the F1 routing.

    cat 5 is a binary "did the model say 'not mentioned'" check; reporting
    BLEU there is meaningless, so we forward the cat-5 binary score.
    cat 1 takes the per-gold max across comma-split sub-answers.
    cat 2/3/4 is plain BLEU after the F1 normalisation + stemming.
    """
    cat = int(category or 0)
    if cat == 5:
        low = (prediction or "").lower()
        return 1.0 if ("not mentioned" in low or "no information available" in low) else 0.0

    pred_norm = _stemmed_norm(prediction)
    if cat == 1:
        preds = [p.strip() for p in (prediction or "").split(",") if p.strip()]
        gts = [g.strip() for g in (ground_truth or "").split(",") if g.strip()]
        if not preds or not gts:
            return 0.0
        pred_norms = [_stemmed_norm(p) for p in preds]
        per_gt = []
        for gt in gts:
            gt_norm = _stemmed_norm(gt)
            per_gt.append(max(_bleu_pair(pn, gt_norm) for pn in pred_norms))
        return sum(per_gt) / len(per_gt)

    if cat == 3:
        ground_truth = (ground_truth or "").split(";")[0].strip()
    return _bleu_pair(pred_norm, _stemmed_norm(ground_truth))


def compute_locomo_metrics(prediction: str, ground_truth: str, category: int) -> Dict[str, float]:
    """Return ``{"f1": ..., "bleu": ...}`` for a single (pred, gt, cat)."""
    return {
        "f1": f1_score(prediction, ground_truth, category),
        "bleu": bleu_score(prediction, ground_truth, category),
    }
