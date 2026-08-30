# LOCOMO split statistics

Split by **conversation**, 2/2/6 (train/val/test), `split_seed=42`, with the
adversarial questions (category 5) removed.

Entry point: `LocomoExperiment.setup_conversation_split(n_train=2, n_val=2, n_test=6, seed=42)`
— what `scripts/run_locomo_percat_evolution.py` and `scripts/eval_topology.py`
call. Pass a different `seed` to resample.

## Set sizes

| split | conversations | QA items | per conversation |
|---|---:|---:|---|
| **train** | 2 | 390 | conv-42=199, conv-48=191 |
| **val** | 2 | 308 | conv-41=152, conv-49=156 |
| **test** | 6 | 842 | conv-26=152, conv-30=81, conv-43=178, conv-44=123, conv-47=150, conv-50=158 |
| **total** | 10 | 1540 | |

By QA count that works out to **25.3% / 20.0% / 54.7%** — not exactly 2:2:6,
because conversations differ in how many questions they carry. By conversation
count it is exactly 2:2:6.

## Category distribution (share within each split in parentheses)

| category | train | val | test | total |
|---|---|---|---|---:|
| 1 multi-hop | 58 (14.9%) | 68 (22.1%) | 156 (18.5%) | 282 |
| 2 temporal | 82 (21.0%) | 60 (19.5%) | 179 (21.3%) | 321 |
| 3 open-domain | 21 (5.4%) | 21 (6.8%) | 54 (6.4%) | 96 |
| 4 single-hop | 229 (58.7%) | 159 (51.6%) | 453 (53.8%) | 841 |
| **total** | **390** | **308** | **842** | **1540** |

## Notes

- The three splits share no conversations, so there is no leakage between them.
- All four categories appear in every split at roughly stable proportions:
  single-hop dominates (~52–59%), open-domain is the thinnest (~5–7%).
- **Category 3 (open-domain) is thin** — 21 items each in train and val, 54 in
  test. Per-category F1 is high-variance there, so read it together with
  `catX_n` rather than on its own.
- Train is only 390 items across 2 conversations (conv-42 and conv-48, the two
  with the most QA), so a skill library can overfit to those two speakers'
  personas and style.
- The adversarial category 5 (446 items) is dropped entirely and takes part in
  no split.
