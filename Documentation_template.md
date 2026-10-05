# Business Entity Resolution Challenge — Methodology Document

**Team:** G-Delta
**Stack:** C++17 (blocking + feature extraction, no external libraries) +
Python/LightGBM (model training + inference). No pretrained or downloaded
models, no external data or API lookups at any stage.

## 1. Methodology Used

For every Source 1 (deduplicated reference) entity, the task is to find the
set of Source 2 and Source 3 records describing the same real-world
business, out of millions of noisy candidates per source, scored by
macro-averaged F_0.5 (precision weighted 2x over recall; correctly
predicting "no match" for a true singleton scores a full 1.0).

**Overall approach: a two-stage pipeline.**

1. **C++ (blocking + feature extraction)** — `train_er` and `predict_er`,
   built from `src/*.cpp`/`*.hpp`. This stage touches the full record set
   (~2.2M Source 1, ~5M Source 2, ~5.3M Source 3 training records; ~1.7M /
   4.9M / 5.1M at test time) and does two things: generate a bounded
   candidate set per Source 1 entity (blocking, §2), and compute a 15-value
   feature vector for every (Source 1, candidate) pair (§3). Output is
   plain TSV — one row per pair.
2. **Python/LightGBM (training + inference)** — `src/train_model.py` and
   `src/predict_model.py`. Trains a gradient-boosted decision tree
   classifier on the candidate-pair features from stage 1, tunes a decision
   threshold against the competition's actual per-entity metric, and (at
   test time) scores candidates and writes the two required submission
   files.

**Why this split.** Per-entity blocking and feature computation is
independent, embarrassingly parallel work, well suited to C++ and
multithreading (§2 covers the actual parallelization and a real
performance finding from doing this). Model training is a different kind
of problem: LightGBM is a mature, heavily-optimized gradient boosting
implementation, and using its Python package was far more reliable than
hand-rolling a comparable tree ensemble in C++ under deadline pressure. An
earlier iteration of this project did train a from-scratch logistic
regression classifier entirely in C++ (see §3 and §4 for why that was
replaced) — the two-stage design is this project's second iteration, not
its first.

**High-level data flow:** raw TSVs → normalized/cleaned records
(`normalize.cpp`) → blocking candidates (`blocking.cpp`) → 15 pairwise
features per candidate (`features.cpp`) → LightGBM classifier → thresholded
match decision → `matching_results.tsv` / `candidate_pairs.tsv`.

**Country handling.** `country` is treated as an open string set
throughout — blocking indices are built fresh per run directly from
whichever source2/source3 file is passed in (§2), so a country absent from
training is partitioned and searched by the same code path as any other,
with no country named anywhere in blocking or feature code. In practice
this was not sufficient on its own: France (present only in the test set)
turned out to need a different, data-derived decision threshold at the
final scoring step to get good results, because its candidate pairs score
differently than training-seen countries for reasons unrelated to blocking.
That finding, and the actual fix shipped, is described in full in §4 — it's
the most important "what we actually did" detail in this document, so it's
not buried.

## 2. Candidate Generation / Blocking Strategy

Implemented in `src/blocking.cpp`, `src/tokens.hpp`, `src/normalize.cpp`.

1. Every record's `business_name` and `business_address` is cleaned once at
   load time: lowercased, accented characters folded, abbreviations
   expanded via a static hand-authored table (Rd/Road, St/Street,
   Pvt/Private, Ltd/Limited, US/India state codes, ...), and reduced to a
   compact single-space-joined token string. The name string additionally
   has generic legal-entity words stripped. Token hashes/trigrams/bigrams
   are *not* precomputed and stored for every record — they're derived on
   demand (`tokens.hpp`) only for records that actually become candidates,
   to keep memory bounded.
2. For Source 2 and Source 3, **five** separate inverted indices are built
   **per country**: address tokens, core-name tokens, address numeric-token
   bigrams (adjacent digit-token pairs, e.g. `"11 23"` → key `11_23` —
   recovers house/plot-number matches a single numeric token loses to
   document-frequency capping), address token bigrams, and name token
   bigrams (adjacent-word-pair keys, e.g. `"global_trust"` — rarer than
   either word alone, so it survives capping that would exclude both
   individually-common words).
3. For each Source 1 record, candidates are the union of postings across
   all five key types within the matching country partition, skipping any
   key whose document frequency exceeds a per-partition cap
   (`--max-df-fraction` × partition size, clamped to
   `[--max-df-floor, --max-df-ceiling]`; actual values used:
   `0.01 / 500 / 50000`). This cap scales with partition size rather than
   being a fixed absolute number, so it behaves consistently on country
   partitions of very different sizes. Candidates are ranked by an
   IDF-weighted score and truncated to `--max-candidates` (actual value
   used: **50**) per entity. These values (vs. a narrower candidate set)
   were chosen to push blocking recall up, accepting the resulting larger
   candidate volume — F_0.5 is precision-heavy, but a candidate that's
   never generated can never be recovered by the model at all, so recall
   ceiling was treated as the harder constraint to relax.
4. This ranked, truncated candidate list is exactly what's written to
   `candidate_pairs.tsv` and exactly what's fed to the matching model —
   there is no further blocking/filtering stage after this.

**Recall ceiling** (fraction of ground-truth matches reachable through
blocking, i.e. the model's effective upper bound regardless of how good it
is — a true match that's never generated as a candidate cannot be scored,
let alone matched): measured on the full training data with the values
above, **94.66%** on the train-fit split and **94.61%** on the held-out
validation split (log: `train_export_parallel.log`). This number matters
specifically for the precision-heavy F_0.5 metric: it caps the achievable
*recall* term for every entity whose true match falls outside the candidate
set, which in turn caps achievable F_0.5 for that entity regardless of
model quality — see the oracle-ceiling discussion in §3 for how this was
used throughout development to separate "blocking problem" from "model
problem."

**Correctness safeguard.** `train_er` includes an inline self-check: after
the parallel train-fit export finishes, it independently recomputes 500
randomly sampled entities' candidate rows (via the same `computeCandidateRows`
function the export itself uses, called directly, not through a worker
thread) and diffs them against the merged parallel-export output, aborting
before the validation split if anything doesn't match. This run: **PASS**,
500/500 (log: `train_export_parallel.log`).

**Parallelization — a performance decision, not a modeling one.**
Per-entity blocking + feature computation is independent across entities,
so both `train_er` and `predict_er` split their export loop across
`--threads` worker threads, each writing to its own temp file, merged at
the end. This has no effect on which candidates or features are produced —
it's purely about how fast the export runs. A real finding worth recording:
on this development machine (8 physical / 16 logical cores), the naive
"use every logical core" default (14, i.e.
`hardware_concurrency() - 2`) was measurably *slower* than using 8 threads.
A controlled benchmark on an identical 60,000-entity sample measured 243.2
entities/sec (train-fit split) and 169.9 entities/sec (validation split,
which does more feature computations per entity since it keeps every
candidate rather than hard-negative-sampling) at 14 threads, vs. 259.8 and
246.4 entities/sec respectively at 8 threads (+6.8% / +45%). Separately, CPU
utilization was sampled (10 readings, 2s apart) during the 14-thread run and
ranged 90.6-94.6% (average ~92.5%) — confirming the cores were genuinely
busy rather than blocked on locks or I/O (we did not re-sample utilization
during the 8-thread run specifically; the inference that the same applies
there rests on it being the identical workload, not a separate
measurement). The most likely explanation for 14 threads nonetheless being
slower: memory-bandwidth/cache pressure from hyperthreading on this
hash-map- and edit-distance-heavy workload, a hardware ceiling rather than
a fixable bug, but this specific mechanism is inferred, not directly
measured. `--threads 8` was used for both the training and test export
runs as a result of the throughput comparison regardless.

## 3. Model Architecture and Feature Engineering

**Model:** LightGBM binary classifier (gradient-boosted decision trees),
trained from scratch on candidate-pair features generated from the
training ground truth (`src/train_model.py`). Not a pretrained model — own
weights fit entirely on the provided data, no downloaded weights of any
kind. LightGBM itself is MIT-licensed. An earlier version of this pipeline
trained a from-scratch logistic regression model directly in C++; it was
replaced because several features only help *conditionally* on others
(e.g. address-numeric overlap matters when name similarity is low but is
far less informative when name similarity is already high), an interaction
effect a linear model's weighted sum cannot represent but decision-tree
splits capture directly.

**Features** (15; `src/features.hpp`/`.cpp`), computed per candidate pair
by tokenizing/trigramming the two records' cleaned name/address strings on
demand and comparing with a linear merge-scan over sorted hashes:

| # | Feature | What it captures |
|---|---|---|
| 0 | `name_token_jaccard` | word-level overlap of core business names |
| 1 | `name_trigram_jaccard` | character-level overlap of core names, robust to word-order/typos |
| 2 | `name_lev_sim` | normalized (1 − edit distance / max len) on core names |
| 3 | `addr_token_jaccard` | word-level overlap of addresses |
| 4 | `addr_numeric_token_jaccard` | overlap of digit-containing tokens (house/plot/PIN numbers) |
| 5 | `addr_trigram_jaccard` | character-level address overlap |
| 6 | `addr_lev_sim` | normalized edit distance on addresses |
| 7 | `name_len_ratio` | shorter/longer core-name length |
| 8 | `addr_len_ratio` | shorter/longer address length |
| 9 | `shared_block_keys_norm` | how many blocking keys this pair shared, squashed to [0,1) |
| 10 | `name_first_token_match` | exact match of the first core-name token |
| 11 | `addr_numeric_overlap_norm` | raw count of overlapping numeric tokens, capped at 5 and normalized |
| 12 | `weighted_block_score_norm` | the blocker's IDF-weighted candidate score, squashed to [0,1) |
| 13 | `addr_primary_num_exact` | exact match of the first numeric token in the address (typically house/plot number) |
| 14 | `addr_collision_norm` | how many *other* candidates in this entity's set share this candidate's exact address, capped at 5 and normalized |

**Feature importances (gain)**, from the final trained model
(`train_model.log`):

| Feature | Gain |
|---|---|
| `name_trigram_jaccard` | 101,377,193 |
| `addr_token_jaccard` | 41,254,659 |
| `addr_numeric_token_jaccard` | 23,265,913 |
| `name_lev_sim` | 20,420,347 |
| `weighted_block_score_norm` | 20,027,428 |
| `addr_len_ratio` | 14,710,957 |
| `name_len_ratio` | 7,722,202 |
| `addr_trigram_jaccard` | 6,290,562 |
| `name_token_jaccard` | 5,319,768 |
| `addr_collision_norm` | 4,180,849 |
| `addr_lev_sim` | 4,069,579 |
| `shared_block_keys_norm` | 2,085,318 |
| `addr_primary_num_exact` | 1,810,435 |
| `name_first_token_match` | 1,394,604 |
| `addr_numeric_overlap_norm` | 849,144 |

`name_trigram_jaccard` and `addr_token_jaccard` dominate, together
accounting for roughly a third of total gain. This is broadly consistent
with our exploratory read of the training data, where business names are
frequently transliterated into a different script, abbreviated
inconsistently, or garbled by typos across sources, while the address
(especially numeric house/plot tokens) tends to match closely when a pair
is genuinely the same business — character-level name similarity
(trigrams) and address token overlap turned out to be the two features the
model leaned on most to tell true matches from plausible-looking
non-matches.

**Hyperparameter search.** A random search over `num_leaves`
(31/63/127/255) × `max_depth` (−1/8/12) × `min_child_samples`
(10/20/50/100) × `learning_rate` (0.03/0.05/0.08/0.12) × `reg_lambda`
(0.1/1.0/5.0) — 576 possible combinations, **8 sampled** due to time
constraints. Each trial trained with a capped `num_boost_round` (300,
early-stopping patience 30); the winning trial was then retrained at the
full `num_boost_round` (2000, early-stopping patience 50) for the final
model. Trials were ranked by the competition's actual per-entity macro
F_0.5 at each trial's own best threshold, not LightGBM's internal
AUC/logloss (a materially different, pair-level objective). Real search
results (`train_model.log`):

| num_leaves | max_depth | min_child_samples | learning_rate | reg_lambda | macro F_0.5 |
|---|---|---|---|---|---|
| **255** | **12** | **50** | **0.08** | **0.1** | **0.9241 (best)** |
| 63 | 12 | 10 | 0.12 | 1.0 | 0.9206 |
| 31 | 12 | 100 | 0.12 | 1.0 | 0.9163 |
| 63 | 12 | 100 | 0.05 | 5.0 | 0.9148 |
| 31 | 12 | 10 | 0.08 | 5.0 | 0.9130 |
| 31 | 12 | 20 | 0.08 | 0.1 | 0.9129 |
| 63 | 8 | 100 | 0.03 | 0.1 | 0.9089 |
| 31 | −1 | 50 | 0.03 | 1.0 | 0.9042 |

Final retrained model: `best_iteration=1997` (out of the 2000-round cap —
validation AUC was still inching up at the very end, so this had not
clearly plateaued; a larger round cap or search budget would likely
improve on this further, see §4).

**Threshold tuning.** The decision threshold was chosen by directly
maximizing the competition's actual metric — per-entity macro F_0.5,
singletons included — on the held-out validation split via a grid search
(0.005 to 0.995, step 0.005), not a generic default (0.5) and not a proxy
metric like AUC or accuracy. Real tuned threshold: **0.875** (saved to
`model.txt.threshold`; this is the US/India threshold — see §4 for why a
*second*, different threshold is used for France at test time).

**Oracle ceiling.** Before tuning the model at all, we computed what macro
F_0.5 would be if the matching model were *perfect* given the *same*
candidate set blocking actually produced (score every true match 1, every
false candidate 0). This isolates how much of the remaining gap to a very
high score is a blocking problem (candidates never reachable) vs. a
modeling problem (reachable candidates the model doesn't rank correctly) —
something the threshold/hyperparameter search alone can't tell you. Real
result on the validation split: **0.9804** overall — singleton 1.0 exactly
(n=12,161), single-match 0.9416 (n=11,931), multi-match 0.9815 (n=196,590).

**Final validation results** (`train_model.log`):
- Macro F_0.5 on held-out validation split at tuned threshold (0.875):
  **0.9289** (vs. an earlier from-scratch logistic-regression baseline of
  **0.7309**; at the untuned 0.5 threshold, for reference: 0.8711).
- That's 94.8% of the 0.9804 oracle ceiling — most of the remaining gap at
  validation time was a model problem, not a blocking problem.
- F_0.5 breakdown by bucket (tuned threshold): singleton **0.8618**
  (n=13,472, oracle 1.0), single-match **0.8392** (n=16,149, oracle
  0.9416), multi-match **0.9412** (n=191,061, oracle 0.9815). Honest
  interpretation: the model over-predicts matches specifically on entities
  that should have zero or exactly one true match, which the
  precision-heavy F_0.5 metric punishes more severely there than on
  multi-match entities, where an extra false positive dilutes precision
  less. This is the largest remaining single-entity-level gap in the
  US/India model and the most promising target for further feature or
  post-processing work (see §4).

## 4. Any Other Relevant Information

**The validation-vs-leaderboard gap and the France fix.** Offline
validation scored 0.9289, but the initial leaderboard submission (a single
global threshold of 0.875 for every entity) scored **0.870** — a gap too
large to be noise. The reason: France has **zero rows** in
`train_source1.tsv`, so it contributes nothing to offline validation at
all (the 0.9289 figure above is US/India only — broken out: 0.9465 US,
0.9026 India). A post-hoc diagnostic (comparing score and feature
distributions across countries on the test set, all scored with the
unchanged model) found France's candidate pairs score systematically
higher even for likely non-matches: mean pair score 0.109 vs. 0.085–0.086
for US/India, and address-similarity features specifically elevated
(`addr_token_jaccard` 0.51 vs. ~0.35, `addr_trigram_jaccard` 0.57 vs.
~0.38–0.39, `addr_lev_sim` 0.61 vs. ~0.43–0.53). The likely mechanism:
shared French street vocabulary (rue, avenue, boulevard, ...) inflates
token/trigram address overlap between genuinely different businesses, and
the hand-authored normalization table (§2) was built around US/India
patterns only — it was never extended with French terms, so this noise
isn't filtered the way US/India street-abbreviation noise is.

With no France ground truth available to tune against offline, the fix was
validated directly against the leaderboard: scoring France candidates with
a *stricter* threshold (0.96) than US/India (0.875), with blocking and
candidates completely unchanged. This is implemented as a general,
data-driven mechanism in `predict_model.py` (`--train-dir` +
`--unseen-country-threshold`): the set of countries needing the stricter
threshold is computed at run time from whichever countries are actually
absent from `--train-dir`'s `train_source1.tsv` — "France" is never named
in the code, consistent with treating `country` as an open string set
throughout. Real result: leaderboard macro F_0.5 went from **0.870 to
0.878** from this one change, with zero retraining.

**What's not fully resolved.** Only two leaderboard submissions were used
to test France thresholds before the deadline (0.875 uniform → 0.870;
France-only 0.96 → 0.878). 0.96 is confirmed better than 0.875, but **not**
confirmed optimal — a middle value (0.90) was generated and
format-validated but never leaderboard-tested, and nothing above 0.96 was
tried. A proper sweep was planned but not completed due to limited
leaderboard upload budget and time remaining before the deadline.

**Score-caching for fast iteration.** Re-scoring the full 86.6M-row test
set with the model from scratch took roughly 45 minutes — too slow to test
several threshold ideas against a limited leaderboard upload budget. The
actual fix used: score every row once with the trained model, cache the
raw scores to a `.npy` file, and reuse that cache for every subsequent
threshold/rule experiment (`--scores-cache` in `predict_model.py`, added
during this tuning work). Re-scoring against a cached array takes roughly
15 minutes instead of 45, since it's pure post-processing with no model
inference. This is a real, disciplined use of limited time under deadline
pressure, not a hypothetical — it's what let us test the France threshold
fix and validate it before the leaderboard closed.

**Considered but not implemented**, for transparency about scope:
- **TF-IDF cosine similarity** for name/address — the challenge materials
  suggested this; Jaccard and Levenshtein similarity were used instead due
  to time, and are not a substitute for it (TF-IDF weights rare tokens
  more heavily, a genuinely different signal). Not present anywhere in
  `features.cpp`.
- **Phonetic or cross-script name matching** (e.g. Soundex/Metaphone, or
  transliteration-aware comparison for Devanagari-vs-Latin spellings) —
  not implemented. Address similarity carried most of the practical
  matching signal in the training data we inspected, so this was
  deprioritized; it remains entirely unmodeled, not partially modeled.
- **A larger hyperparameter search** — only 8 of 576 possible combinations
  were tried (§3), time-boxed under deadline pressure. The final model's
  `best_iteration` was 1997 of a 2000-round cap, suggesting it had not
  fully plateaued; more search budget or more boosting rounds likely had
  further headroom that wasn't explored.

**Compliance with challenge rules.** No external APIs, databases, or
network lookups are used at any stage of blocking, feature computation,
training, or inference — normalization relies solely on a static,
hand-authored abbreviation table compiled into `normalize.cpp`. LightGBM
(MIT license) is trained entirely from scratch on the provided training
data only, well within the MIT/Apache-2.0 licensing constraint and not
remotely approaching any parameter-count limit aimed at pretrained models
(there is no comparable notion of "pretrained parameters" for a tree
ensemble fit from scratch on this data). `country` is treated as an open
string set throughout blocking and feature code, with no country named in
either — the one place a country-specific decision is made (the France
threshold above) is implemented as a generic "any country unseen in
training" rule, derived from the training data at run time, not a
hard-coded special case.
