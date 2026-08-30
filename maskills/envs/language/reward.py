"""Verified reward computation for language task episodes.

- QA / Math: LLM-as-judge comparing model output against ground truth (0 or 1).
- Coding: HumanEval execution-based evaluation (0 or 1).
"""

from __future__ import annotations

import re
from typing import Tuple

from maskills.core.base import BaseReward, Trajectory

from .metrics import compute_qa_metrics

try:
    from human_eval.execution import check_correctness as _humaneval_check_correctness
    _HUMANEVAL_AVAILABLE = True
except ImportError:
    _HUMANEVAL_AVAILABLE = False


# ── LLM-as-judge prompt templates ────────────────────────────────────

_QA_SYSTEM = (
    "You are a precise and objective judge for question-answering tasks. "
    "Follow the output format exactly."
)

_QA_USER = """\
Question: {question}

Ground Truth Answer: {ground_truth}

Model's Answer:
{response}

Is the model's answer CORRECT?
Accept paraphrasing, synonyms, and different but equivalent formulations.
The model's answer may include explanations, reasoning, or extra commentary —
ignore these and judge only by whether the final answer it gives matches the
ground truth (semantically). Length, style, or extra context do NOT make a
correct answer wrong.

Respond EXACTLY in this format (nothing else):

VERDICT: CORRECT
or
VERDICT: INCORRECT
"""

_MATH_SYSTEM = (
    "You are a precise mathematical judge. "
    "Follow the output format exactly."
)

_MATH_USER = """\
Math Problem:
{problem}

Correct Answer: {ground_truth}

Model's Solution:
{response}

Is the model's final numerical/symbolic answer mathematically equivalent to the correct answer?
Ignore differences in formatting (e.g. fractions vs decimals, \\boxed notation).
The model's solution may include step-by-step reasoning, derivations, or extra
commentary — ignore these and judge only by whether the final answer it
produces is mathematically equivalent to the ground truth.

Respond EXACTLY in this format (nothing else):

VERDICT: CORRECT
or
VERDICT: INCORRECT
"""


# ── Reward generator ─────────────────────────────────────────────────

class VerifiedRewardGenerator(BaseReward):
    """Compute verified rewards: LLM-as-judge for QA/Math, execution for Coding."""

    def __init__(self, judge_model: str = "openai/gpt-5.1", code_timeout: float = 10.0,
                 qa_reward_metric: str = "f1"):
        """
        Args:
            qa_reward_metric: how to score QA trajectories — ``"f1"`` (token
                F1, the default optimization target; precision-sensitive, so
                it penalises verbose answers a lenient judge would accept),
                ``"em"`` (exact match), or ``"judge"`` (binary LLM-as-judge).
        """
        self.judge_model = judge_model
        self.code_timeout = code_timeout
        self.qa_reward_metric = qa_reward_metric
        self._client = None  # OpenAI-compatible client, injected via set_client

    def set_client(self, client):
        """Inject an OpenAI-compatible client for LLM-as-judge calls."""
        self._client = client

    def compute(self, trajectory: Trajectory) -> float:
        task_type = trajectory.metadata.get("task_type", "")
        task = trajectory.task
        final_answer = trajectory.metadata.get("final_answer", "")
        if not final_answer and trajectory.steps:
            final_answer = trajectory.steps[-1].get("output", trajectory.steps[-1].get("action", ""))

        if task_type == "qa":
            # Token F1/EM/precision/recall/BLEU from the upstream HotpotQA
            # evaluator — always computed (cheap, no LLM).
            metrics = {}
            try:
                metrics = compute_qa_metrics(
                    final_answer or "", task.get("ground_truth", ""),
                )
                trajectory.metadata["qa_metrics"] = metrics
            except Exception:  # metrics never fail the run
                pass
            if self.qa_reward_metric == "judge":
                # Binary LLM-as-judge correctness as the training signal.
                score, _ = self._judge_qa(final_answer, task)
            else:
                # Reward IS the token metric (default: F1) — the optimization
                # target.  Unlike the lenient judge it is precision-sensitive,
                # so verbose / low-precision answers score low.
                score = float(metrics.get(self.qa_reward_metric,
                                          metrics.get("f1", 0.0)))
        elif task_type == "math":
            score, _ = self._judge_math(final_answer, task)
        elif task_type == "coding":
            score, _ = self._exec_coding(final_answer, task)
        else:
            score = trajectory.reward
        return score

    # ------------------------------------------------------------------
    # QA
    # ------------------------------------------------------------------

    def _judge_qa(self, response: str, task: dict) -> Tuple[float, str]:
        # Gold context is intentionally NOT passed to the judge: HotPotQA's
        # context is itself imperfect / can disagree with the answer string,
        # and showing it to the judge biases verdicts away from semantically
        # correct answers that paraphrase the gold answer.
        question = task.get("question", "")
        ground_truth = task.get("ground_truth", "")
        prompt = _QA_USER.format(
            question=question,
            ground_truth=ground_truth,
            response=response,
        )
        return self._call_judge(prompt, system=_QA_SYSTEM)

    # ------------------------------------------------------------------
    # Math
    # ------------------------------------------------------------------

    def _judge_math(self, response: str, task: dict) -> Tuple[float, str]:
        problem = task.get("question", "")
        ground_truth = task.get("ground_truth", "")
        prompt = _MATH_USER.format(
            problem=problem,
            ground_truth=ground_truth,
            response=response,
        )
        return self._call_judge(prompt, system=_MATH_SYSTEM)

    # ------------------------------------------------------------------
    # Coding (HumanEval execution)
    # ------------------------------------------------------------------

    def _exec_coding(self, response: str, task: dict) -> Tuple[float, str]:
        if not _HUMANEVAL_AVAILABLE:
            raise RuntimeError(
                "human_eval package is required for coding evaluation. "
                "Install it with: pip install human-eval"
            )

        test_code = task.get("test", "")
        if not test_code:
            return 0.0, "No test cases provided."

        entry_point = task.get("entry_point", "")
        prompt = task.get("question", "")

        # Extract code block from markdown-fenced response
        match = re.search(r"```(?:python)?[ \t]*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        completion = match.group(1) if match else response

        problem = {
            "task_id": task.get("task_id", "unknown"),
            "prompt": prompt,
            "test": test_code,
            "entry_point": entry_point,
        }
        result = _humaneval_check_correctness(problem, completion, timeout=self.code_timeout)
        if result["passed"]:
            return 1.0, "Passed"
        return 0.0, f"Failed: {result['result']}"

    # ------------------------------------------------------------------
    # Shared judge helper
    # ------------------------------------------------------------------

    def _call_judge(self, prompt: str, system: str = "") -> Tuple[float, str]:
        if self._client is None:
            raise RuntimeError("No LLM client set. Call set_client() before evaluation.")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = self._client.chat.completions.create(
            model=self.judge_model,
            messages=messages,
            max_tokens=64,
        )
        text = result.choices[0].message.content.strip()
        if re.search(r"\bCORRECT\b", text) and not re.search(r"\bINCORRECT\b", text):
            return 1.0, text
        return 0.0, text
