<div align="center">

# MASkills: Continual Skills Optimization for Multi-Agent LLM Systems

[![CI](https://github.com/DaRL-GenAI/MASkills/actions/workflows/ci.yml/badge.svg)](https://github.com/DaRL-GenAI/MASkills/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026%20Findings-blueviolet?style=flat-square)](https://github.com/DaRL-GenAI/MASkills)

<p align="center">
  <a href="https://github.com/DaRL-GenAI/MASkills"><img src="https://img.shields.io/github/stars/DaRL-GenAI/MASkills?style=for-the-badge&logo=github&logoColor=white&color=red" alt="GitHub Stars"></a>
</p>

[**Quick Start**](#quick-start) · [**Key Components**](#key-components) · [**Benchmarks**](#benchmarks) · [**Skill Format**](#skill-format) · [**Citation**](#citation) · [**FAQ**](#faq)

</div>

An LLM agent that learns from experience usually keeps what it has learned in one system prompt. Every update rewrites that prompt as a whole, so an edit aimed at one failure mode silently perturbs unrelated competence, and no single piece of knowledge can be credited, revised, or retired on its own. MASkills makes that knowledge **discrete and persistent** -- a library of individually addressable sub-skills -- and evolves it with typed operators under per-skill credit assignment, so an edit is aimed at the one skill the evidence implicates and is rolled back if it does not hold up.

---

### News

> **[Aug 2026]** MASkills v1.0.0 released -- three benchmarks (GAIA, HotpotQA, LOCOMO) behind six entry points, each with a settable agent topology and skill-library initialization.
>
> **[Aug 2026]** Paper accepted to **EMNLP 2026 Findings**.

<!-- TODO: add the arXiv badge and link once the preprint is posted, here and in CITATION.cff. -->

---

## Overview

MASkills replaces the monolithic prompt with a **skill library**: a directory of `SKILL.md` files, each an independently addressable unit of task knowledge. Training acts on that structure rather than on one string.

| Monolithic prompt optimization | MASkills |
|---|---|
| One prompt string | Skill library `K = {k_1, ..., k_m}`, each skill separately addressable |
| An update rewrites the whole prompt | Typed operators act on one skill at a time |
| Credit lands on the agent | Credit lands on the skills the trajectory actually invoked |
| The prompt grows monotonically | **Consolidate** merges overlaps, **prune** retires dead weight |
| A bad update is discovered later, if at all | A held-out validation gate rolls the candidate library back |
| Knowledge is opaque | Every skill is a readable file with provenance and measured utility |

Four typed operators:

| Operator | Effect |
|---|---|
| **Induct** | Propose a new sub-skill from a recurring failure pattern |
| **Refine** | Edit one existing sub-skill, `k <- k + dk` |
| **Consolidate** | Merge redundant or overlapping sub-skills |
| **Prune** | Remove a sub-skill whose measured utility is low |

---

## Features

| Feature | Description |
|---------|-------------|
| **Discrete Skill Library** | Task knowledge as `SKILL.md` files in the [Anthropic Agent Skills format](https://agentskills.io/specification), readable and editable by hand |
| **Four Typed Operators** | Induct, refine, consolidate and prune, each on its own cadence |
| **Per-Skill Credit** | An episode's outcome is attributed to the skills its trajectory invoked, not to the agent as a whole |
| **Validation Gate** | A candidate library is committed only if held-out reward does not regress beyond a tolerance |
| **Three Benchmarks** | GAIA (tool use), HotpotQA (multi-hop QA), LOCOMO (long-term conversational memory) |
| **Three Topologies** | `decentralized`, `centralized` and `hybrid`, settable on both training and evaluation |
| **Optional Role Evolution** | The agent's role prompt can evolve alongside its skills, or stay fixed (the default) |
| **Single or Paired Agents** | One library per agent, co-evolved -- retriever + reasoner, Researcher + Solver |
| **Resumable Evaluation** | Results append per task id, so an interrupted sweep re-runs only what is missing |

---

## Quick Start

### Installation

```bash
git clone https://github.com/DaRL-GenAI/MASkills.git
cd MASkills

pip install -e .                  # core library
pip install -e ".[gaia]"          # + GAIA attachment parsing (xlsx / pdf / docx / pptx)
pip install -e ".[language]"      # + HotpotQA / MATH / HumanEval metrics and tools
pip install -e ".[all]"           # everything
```

Set your API key -- never hardcode it in a script:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="..."      # your provider's endpoint; defaults to OpenRouter
export TAVILY_API_KEY="..."       # only for GAIA's SEARCH and BROWSE tools
```

Benchmark data is not redistributed here. Fetch it into the paths the scripts expect:

- **GAIA** -- gated on Hugging Face (`gaia-benchmark/GAIA`). Put the split JSONL files under `data/gaia/`, one JSON object per line with at least `task_id`, `Question` and `Final answer`.
- **HotpotQA** -- under `env/lang_benchmark/HotPotQA/`.
- **LOCOMO** -- clone [snap-research/LoCoMo](https://github.com/snap-research/LoCoMo) into `env/locomo/`; the loader reads `env/locomo/data/locomo10.json`. See `configs/locomo/SPLIT_STATS.md` for the split.

### Minimal Example

```python
import maskills

config = maskills.LanguageTaskConfig(
    task_type="qa",
    paradigm="central_credit",
    skill_evolution=True,
    llm=maskills.LLMConfig.from_preset("gpt-4o-mini"),
)

env = maskills.make_env("language", config)
trainer = maskills.SkillEvolutionTrainer(
    config=config,
    env=env,
    critic=maskills.SkillCreditCritic(config),
    optimizer=maskills.SkillEvolutionOptimizer(config.get_optimizer_llm()),
)
trainer.train()
```

Or train from a config file in one line:

```python
maskills.train("configs/language_task/qa_hotpot_decentralized.json")
```

### CLI Usage

Six entry points: train and evaluate, for each benchmark. Every one takes `--topology` and a skill-library flag, so a library can be trained under one topology and scored under another.

| | GAIA | HotpotQA | LOCOMO |
|---|---|---|---|
| **train** | `scripts/train_gaia.py` | `scripts/train_hotpotqa.py` | `scripts/train_locomo.py` |
| **evaluate** | `scripts/eval_gaia.py` | `scripts/eval_hotpotqa.py` | `scripts/eval_locomo.py` |

```bash
# HotpotQA and LOCOMO: one invocation runs every iteration.
python scripts/train_hotpotqa.py --n-train 100 --n-val 50 --n-test 100 --iters 5
python scripts/train_locomo.py --category 1 --iters 5   # 1=multihop 2=temporal 3=opendomain 4=singlehop

# GAIA keeps its library on disk, so one invocation is one iteration.
python scripts/train_gaia.py --topology decentralized --iter 1 \
  --init-skills <K_i dir> --out-skills <K_i+1 dir>

# Evaluate. --skills empty is the no-skills floor each benchmark reports.
python scripts/eval_hotpotqa.py --topology hybrid --skills <library dir>
python scripts/eval_locomo.py --category 1 --skills empty
python scripts/eval_gaia.py --skills empty --limit 20
```

Every trainer also accepts `--init-skills empty` to start from no skills at all, and `--ablation` to disable one mechanism at a time.

---

## Key Components

MASkills adds four components on top of a language-space CTDE loop:

1. **Skill Library** -- Each agent holds a set of discrete skills `K_i = {k_1, ..., k_m}`, each with a description used for routing, a body the agent reads, and a running utility estimate.

2. **Skill-Conditioned Credit** -- The critic sees the trajectory *and* the skills it invoked, and attributes the outcome per skill, plus a residual for what no skill covers.

3. **Typed Operators** -- Induct, refine, consolidate and prune, each on its own cadence. Refinement is cheap and runs often; induction, consolidation and pruning act on longer horizons, once evidence has accumulated.

4. **Validation Gate** -- The candidate library is rolled out on a held-out split and committed only if its reward stays within a tolerance of the current one. Otherwise the current library is restored, so a bad iteration costs time rather than competence.

One iteration:

```
Rollout under K_i -> per-skill credit + residual -> aggregate to a gradient per skill
  -> apply the scheduled operators -> candidate K_i+1 -> validation gate -> commit or roll back
```

### Library API

| Class | Description |
|-------|-------------|
| `maskills.SkillLibrary` / `maskills.Skill` | The discrete library and its units |
| `maskills.SkillCreditCritic` | Critic performing per-skill causal attribution |
| `maskills.SkillEvolutionOptimizer` | The four operators over a library |
| `maskills.SkillEvolutionTrainer` | Training loop with the operator schedule and the validation gate |
| `maskills.skill_lib` | On-disk `SKILL.md` library I/O (`load_lib`, `apply_ops`, `snapshot_lib`) |
| `maskills.LLMConfig` | Unified LLM backend configuration |
| `maskills.make_env` / `maskills.register_env` | Environment registry |

---

## Benchmarks

| Benchmark | Agents | Tools | What the skills have to learn |
|---|---|---|---|
| **GAIA** | 1, or Researcher + Solver | SEARCH / BROWSE (Tavily), COMPUTE (sandboxed Python), attachment parsing | Tool-call syntax, evidence discipline, the exact-match answer format |
| **HotpotQA** | 2, sequential | Wiki search | Multi-hop decomposition, keeping the final answer minimal |
| **LOCOMO** | Retriever + Reasoner | grep over the conversation | A grep plan per hop, carrying session dates through the handoff, the grader's answer shape |

LOCOMO is additionally trained per question category, because a good skill for a temporal question looks nothing like one for a multi-hop question.

### Topologies

| Topology | Shape |
|---|---|
| `decentralized` | Sequential: `agent_1` -> `agent_2`, sharing a message pool |
| `centralized` | A main agent delegates via `<retrieve>`; each sub-call runs in a fresh context |
| `hybrid` | Same delegation, but the sub-agent accumulates state across calls |

Any library can be scored under any topology, which is what makes the comparison controlled:

```bash
for topo in decentralized centralized hybrid; do
  python scripts/eval_hotpotqa.py --topology $topo --skills <library dir>
done
```

GAIA is the exception, because it has two rollout implementations. `--engine library` (the default) is the on-disk pipeline the trainer uses, so its score is comparable to training, and it implements centralized and decentralized. `--engine env` drives the agents through `GaiaEnv`, the path the other benchmarks use, and adds hybrid -- so a topology comparison on GAIA needs that engine for all three.

---

## Skill Format

A skill library is a directory of `SKILL.md` files in the [Anthropic Agent Skills format](https://agentskills.io/specification):

```
<lib_dir>/
    SKILL.md            # root identity / always-on instructions
    <skill_name>/
        SKILL.md        # one sub-skill, loaded and edited independently
    ...
```

Each `SKILL.md` is YAML frontmatter followed by a markdown body:

```markdown
---
name: cross-verify-fact
description: Re-derive a numeric or factual answer via a second independent method before finalizing.
---

Before emitting FINAL ANSWER, recompute the result using a different approach
(unit conversion, alternate source, or a sanity-check calculation). If the two
disagree, investigate before answering.
```

`description` is what the agent sees when deciding whether to load a skill, so it carries the routing signal; the body is what it reads once it does.

Two paths consume a library. **On disk** (GAIA), `config.agent_skills_dirs` points an agent at a directory and `maskills.skill_lib` applies the optimizer's operations back to it. **In memory** (HotpotQA, LOCOMO), `load_skills_dir` reads it into a `SkillLibrary` that lives inside the policy and is evolved by `SkillEvolutionTrainer`.

Setting `skills_dir` on a config instead installs a **fixed** library: rendered into every agent's prompt, but the optimizer never sees it and no operator touches it.

### Evolving the role as well as the skills

An agent's prompt has two parts: its **role** -- where it sits in the team and how it must communicate -- and its **skill library**. By default training moves only the library, because the role carries the collaboration protocol the environment parses (the handoff format, the tags the agent emits, which agent speaks last), and an edit there can invalidate every trajectory at once in a way no skill can repair.

`--evolve-role` on any trainer turns it on. What drives an edit is the *residual*: the failures credit assignment could not attribute to any skill, which is exactly the part of the prompt the library does not cover. Edits are anchor-based rather than a rewrite, and the proposed role goes through the same validation gate as the library.

```bash
python scripts/train_hotpotqa.py --evolve-role --iters 5
```

On GAIA the role is the library's root `SKILL.md`, edited by a `refine` operation on the `_root_` slug.

---

## Configuration

Skill evolution is driven by a handful of config fields, settable in JSON or as `overrides`:

| Field | Default | Meaning |
|---|---|---|
| `skill_evolution` | `False` | Use the operators instead of a monolithic prompt rewrite |
| `skill_eval_delta` | `0.02` | Validation gate slack; a candidate is kept while held-out reward stays within it |
| `refine_every` | `1` | Cadence of each operator, in iterations. `0` disables that operator |
| `induct_every` | `2` | |
| `consolidate_every` | `3` | |
| `prune_every` | `3` | |
| `max_skills_per_agent` | `12` | Induction will not grow a library past this |
| `hard_trajectory_threshold` | `0.5` | Reward at or below which a trajectory counts as induction evidence |
| `evolve_role` | `False` | Whether the role prompt evolves alongside the skills |
| `architecture` | `decentralized` | Agent topology |

`MASKILLS_LLM_TIMEOUT`, `MASKILLS_LLM_MAX_RETRIES` and `MASKILLS_LLM_RETRY_BACKOFF` tune the client's retry behaviour.

---

## Extending to a New Benchmark

Subclass `maskills.BaseEnvironment`, implement `reset`, `step`, `sample_tasks` and `collect_trajectory`, and register it:

```python
from maskills.envs import register_env

@register_env("my_task")
class MyEnv(maskills.BaseEnvironment):
    ...
```

To make it skill-aware, render each agent's library into its system prompt with `render_skill_library`, and record which skills each turn invoked with `build_skill_trace` -- that trace is what lets the critic attribute credit per skill rather than per agent.

---

## Project Structure

```
maskills/                  # The library (pip-installable)
├── core/                  # Skill library, per-skill credit, the four operators,
│                          #   base classes, critic, policy, trajectory
├── config/                # Unified configuration system
├── trainer/               # SkillEvolutionTrainer, MonteCarloTrainer, callbacks
├── experiments/           # Per-benchmark training and evaluation logic
├── envs/                  # Environment registry and the three benchmarks
│   ├── gaia/              #   Tool stack, rollouts, task loader
│   ├── language/          #   HotpotQA / MATH / HumanEval, search tools
│   └── locomo/            #   Long-term conversational memory, grep tool
├── llm/                   # LLM client and token tracking
├── store/                 # Checkpointing, trajectory storage, logging
└── cli/                   # `maskills train` entry point

scripts/                   # Six entry points: train and evaluate x three benchmarks
configs/                   # Experiment configs, one per benchmark × paradigm
tests/                     # Test suite (pytest)
```

---

## FAQ

<details>
<summary><b>How do I configure the API key?</b></summary>

**Option 1**: Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export TAVILY_API_KEY="tvly-..."   # GAIA only
```

**Option 2**: Point `api_key_env_var` inside the config's `llm` object at a different environment variable.

Never put a literal key in a tracked file.

</details>

<details>
<summary><b>Which LLM backends work?</b></summary>

Any OpenAI-compatible endpoint. Model names are given in `provider/model` form (`openai/gpt-4o`, `qwen/qwen-2.5-7b-instruct`) against the `base_url` you set, so OpenRouter, the OpenAI API, or any compatible gateway all work. `maskills.list_available_models()` lists the presets.

</details>

<details>
<summary><b>What does <code>--skills empty</code> do?</b></summary>

It evaluates the agents with no task knowledge -- the floor each benchmark's table reports alongside the trained libraries. The same idea on a trainer (`--init-skills empty`) starts training from nothing, which tests whether the operators can build a useful library from scratch. It is also the default everywhere, since no library ships with the repository.

On HotpotQA and LOCOMO the library is literally empty. On GAIA it is *protocol only*: the tool syntax (`SEARCH:`, `BROWSE:`, `COMPUTE:`) and the output contract (`HANDOFF_TO_SOLVER`, `FINAL ANSWER:`), with no task knowledge of any kind. A GAIA agent given a blank prompt would fail on the wire format rather than on the task, which would measure the wrong thing; see `maskills/envs/gaia/protocol.py`.

</details>

<details>
<summary><b>How do I resume an interrupted evaluation?</b></summary>

Re-run the same command. Results append to `<tag>_rows.jsonl` keyed by task id, and a re-run skips the tasks already there. `--report-only` re-summarizes what is on disk without running anything.

</details>

<details>
<summary><b>Why does GAIA have two evaluation engines?</b></summary>

GAIA has two rollout implementations: the on-disk `SKILL.md` pipeline the trainer uses, and `GaiaEnv`, which drives the agents the way the other benchmarks do and is the only one implementing hybrid. Their numbers are not interchangeable, so `--engine` makes the choice explicit rather than mixing them silently.

</details>

<details>
<summary><b>What does each ablation disable?</b></summary>

- `credit` -- the optimizer is shown every rollout rather than only the failures, removing the per-task credit signal.
- `rollback` -- the validation gate never rejects, so every candidate library is committed.
- `consolprune` -- only induct and refine run, so the library can grow but never contract.
- `momentum` (LOCOMO) -- skill utility is overwritten with the batch mean instead of an EMA.

</details>

---

## Citation

If MASkills is useful in your research, please cite:

```bibtex
@inproceedings{yao2026maskills,
  title     = {MASkills: Continual Skills Optimization for Multi-Agent LLM Systems},
  author    = {Yao, Huaiyuan and Liu, Xiaoou and Fleming, Charles and Chen, Tianlong and Wei, Hua},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

---

## Acknowledgements

MASkills builds on [LangMARL](https://github.com/DaRL-GenAI/LangMARL) for its language-space CTDE substrate, and on the [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA), [HotpotQA](https://hotpotqa.github.io/) and [LOCOMO](https://github.com/snap-research/LoCoMo) benchmarks. Each keeps its own license.

---

## License

Released under the [MIT License](LICENSE).
