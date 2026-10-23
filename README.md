# Business Entity Resolution

An entity-resolution pipeline that matches noisy business records (name +
address + country) across three independent data sources — built for the
**Business Entity Resolution Challenge**, hosted on the Unstop platform.
Given a deduplicated reference set (Source 1) and two noisy source sets
(Source 2, Source 3), it finds every record across sources that describes
the same real-world business, scored by macro-averaged F_0.5 (precision
weighted 2x over recall; correctly predicting "no match" for a true
singleton scores a full 1.0).

**Result: 0.878 macro F_0.5** on the competition leaderboard, up from a
0.7309 from-scratch logistic-regression baseline. See
[`Documentation_template.md`](Documentation_template.md) for the formal
methodology write-up (the challenge's required deliverable).

## What you can use this for

The competition dataset isn't public (see "Data source" below), so this
isn't a clone-and-run project for most visitors — but the pipeline itself
is dataset-agnostic and the individual pieces are reusable on their own:

- **As a working template for a record-matching / deduplication problem**
  across multiple noisy sources: point `--train-dir`/`--test-dir` at your
  own data in the same three-source, four-column shape (`entity_id`,
  `name`, `address`, `country` → adjust field names to taste) and the
  blocking + feature + LightGBM pipeline runs as-is. `country` is never
  hardcoded, so this works for any number of countries/regions, seen or
  unseen in training.
- **As a reference for the individual techniques**, independent of the
  dataset: per-country inverted-index blocking over multiple key types
  (tokens, trigrams, numeric bigrams); a hand-built similarity feature set
  for name/address matching; tuning a decision threshold by directly
  maximizing the *actual* target metric instead of a proxy like AUC; the
  "oracle ceiling" method for telling whether a low score is a candidate-
  generation problem or a model problem; and the score-caching pattern for
  iterating on post-processing rules quickly when re-scoring from scratch
  is expensive (see "Fast iteration" below).
- **As a worked example of a hybrid C++/Python pipeline** — performance-
  critical, embarrassingly-parallel work (blocking, feature extraction
  over millions of records) in C++, model training/tuning in Python where
  the ML ecosystem (LightGBM) is stronger — including a real, measured
  finding on why naive "use all logical cores" threading can be slower
  than expected on hyperthreaded hardware (see "Threading note" below).

## How it works

A **two-stage pipeline**:

1. **C++ (blocking + feature extraction)** — `src/train_main.cpp` /
   `src/predict_main.cpp`. Builds five per-country inverted indices
   (address tokens, name tokens, address numeric-token bigrams, address
   token bigrams, name token bigrams) from whichever source2/source3 file
   is passed in, so an unseen country (France, in the test set) is handled
   by the same code path as any other — no country is ever named in
   blocking or feature code. For each Source 1 entity this generates a
   bounded candidate set (blocking), then computes 15 hand-built pairwise
   similarity features per candidate (name/address token and trigram
   Jaccard, bounded Levenshtein similarity, numeric house/plot-number
   overlap, length ratios, first-token match, blocking-key-overlap
   signals, an address-collision signal). Pure C++17, no external
   libraries — this is the only stage that touches the full record set
   (millions of rows per source), so it's parallelized across worker
   threads, each handling an independent slice of entities and writing to
   its own temp file, merged at the end. Output is plain TSV: one row per
   candidate pair, feature columns, and (for training) a 1/0 label from
   the ground truth.
2. **Python/LightGBM (training + inference)** — `src/train_model.py` /
   `src/predict_model.py`. Trains a LightGBM binary classifier on those
   candidate-pair features, using a random hyperparameter search
   evaluated by the competition's *actual* metric — per-entity macro
   F_0.5, singletons included — not AUC/logloss, then tunes the decision
   threshold the same way. `predict_model.py` loads the saved model +
   threshold and scores the test-set candidate pairs (streamed in chunks,
   so memory stays bounded) to produce the two required submission files.

This split exists because several features only help *conditionally* on
others — an interaction effect plain logistic regression (an earlier,
single-stage version of this pipeline) couldn't exploit, but LightGBM's
tree splits can.

**Why address matters, but not as much as expected.** Exploring the
training data shows many true matches where the business name is
transliterated into a different script, garbled by typos, or unrelated
between sources, while the address (especially numeric house/plot tokens)
matches almost exactly — so blocking leans on address. But the trained
model's feature-importance ranking (below) shows name-similarity features
mattered *more* than that initial read suggested: LightGBM's tree splits
pick up name-vs-address interactions a linear model couldn't.

## Data source

The dataset is **not included in this repository, and there is no public
link to it.** It's several GB of competition data that was distributed
privately to registered participants of the Business Entity Resolution
Challenge (now concluded) — it isn't ours to redistribute, and no public
mirror exists. If you weren't a participant, treat this repo as a code and
methodology reference, not a clone-and-run project: you won't be able to
reproduce the exact numbers below without that data, though the pipeline
itself works against any dataset in the same shape (see below). If you do
have a copy — a challenge participant, or you're supplying your own
similarly-structured data — arrange it as:

```
dataset/
├── train/
│   ├── train_source1.tsv       # deduplicated reference entities
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv  # source1_entity_id -> comma-separated matched ids
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Each source file has columns `entity_id, business_name, business_address,
country`. If you're a participant in this (or a similarly-shaped)
challenge, get the data from the platform you registered on. If you're
adapting this code to your own entity-resolution problem, point
`--train-dir`/`--test-dir` (below) at your own data in this same shape —
the pipeline never hardcodes a country list or any dataset-specific
assumption, by design.

## Build

Requires a C++17 compiler and CMake. Tested with g++ 16.2 (MSYS2 MinGW-w64)
and CMake 4.4. Run from the repo root (the directory containing
`dataset/` — same place every command below runs from):

```sh
cmake -S code/business_entity_resolution -B code/business_entity_resolution/build -DCMAKE_BUILD_TYPE=Release
cmake --build code/business_entity_resolution/build -j
```

Produces `code/business_entity_resolution/build/train_er` and `.../predict_er`
(`.exe` on Windows, referenced below without the extension). Both link
against the MinGW runtime (libstdc++/libgcc/libwinpthread) on Windows —
make sure `<mingw64>/bin` is on `PATH` when *running* them, not just when
building, or they'll fail to start with a DLL-not-found error.

```sh
pip install -r code/business_entity_resolution/requirements.txt   # lightgbm, pandas, numpy (pinned)
```

No network access is needed at train/inference time.

## Run

Commands below use `python3`; on Windows, if `python3` resolves to the
Microsoft Store stub instead of a real interpreter (it prints an
install-from-Store prompt and does nothing), use the `py -3` launcher
instead — that's what was actually used to produce the results below.

```sh
# 1. C++: blocking + feature export for train-fit and validation splits.
# --threads: pass your physical core count explicitly -- see "Threading
# note" below for why the naive "every logical core" default can be
# measurably SLOWER on hyperthreaded CPUs.
./code/business_entity_resolution/build/train_er \
    --train-dir dataset/train --threads 8

# 2. Python: train LightGBM (hyperparameter search + threshold tuning), save the model
python3 code/business_entity_resolution/src/train_model.py \
    --train-features code/business_entity_resolution/artifacts/train_features.tsv \
    --val-features code/business_entity_resolution/artifacts/val_features.tsv \
    --val-entities code/business_entity_resolution/artifacts/val_entities.tsv \
    --model-out code/business_entity_resolution/artifacts/model.txt

# 3. C++: blocking + feature export for the test set
./code/business_entity_resolution/build/predict_er \
    --test-dir dataset/test --threads 8

# 4. Python: score test candidates, write the two required output files.
# --train-dir + --unseen-country-threshold: a stricter threshold for any
# country absent from --train-dir (e.g. France here) -- see "France-specific
# tuning" below for why. Omit both flags to fall back to one global
# threshold for every country.
python3 code/business_entity_resolution/src/predict_model.py \
    --test-dir dataset/test \
    --train-dir dataset/train \
    --unseen-country-threshold 0.96 \
    --features code/business_entity_resolution/artifacts/test_features.tsv \
    --model code/business_entity_resolution/artifacts/model.txt \
    --matching-out output/matching_results.tsv \
    --candidate-out output/candidate_pairs.tsv

# 5. Validate the output format.
# utils/validate_submission.py ships with the challenge's own project
# skeleton (same place as dataset/) -- stdlib-only, run it from wherever
# your copy of the skeleton lives.
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The C++ export step is by far the most expensive part of this pipeline —
budget roughly 1-2 hours per full export on a modern multi-core machine at
the dataset's full scale (millions of records per source). Use
`code/business_entity_resolution/src/make_dev_subset.py` to build a small
stratified subset for fast local iteration instead of running against the
full dataset every time.

`train_er` prints, per run: thread count, training-set size and class
balance, blocking recall ceiling on both splits, an inline self-check
(recomputes ~500 random train-fit entities independently and diffs them
against the merged parallel-export output — aborts before the validation
split if anything doesn't match), and the bucketed (singleton/single-match
/multi-match) oracle ceiling. `train_model.py` prints the hyperparameter
search trials, best config, tuned threshold, final macro F_0.5, feature
importances, and the same bucketed F_0.5 breakdown.

### CLI flags

`train_er`: `--train-dir`, `--train-features-out`, `--val-features-out`,
`--val-entities-out`, `--val-fraction` (default 0.10), `--max-df-fraction`
(default 0.01), `--max-df-floor` (default 500), `--max-df-ceiling` (default
50000), `--max-candidates` (default 50), `--neg-min` (default 5),
`--neg-multiplier` (default 3), `--seed`, `--threads` (default: auto),
`--limit-entities` (debug: caps entity counts for quick correctness checks),
`--dump-examples` (dumps blocking-miss false negatives for inspection).

`predict_er`: `--test-dir`, `--features-out`, `--max-df-fraction`,
`--max-df-floor`, `--max-df-ceiling`, `--max-candidates` (should match the
values used for training), `--threads`.

`train_model.py`: `--train-features`, `--val-features`, `--val-entities`,
`--model-out`, `--search-iterations` (default 12; 0 disables the search and
uses `--num-leaves`/`--learning-rate`/etc. directly), `--search-boost-round`
(default 300, capped rounds used per search trial), `--num-boost-round`
(default 2000, used for the final model once the best config is found),
`--early-stopping-rounds`, `--seed`.

`predict_model.py`: `--test-dir`, `--features`, `--model`, `--matching-out`,
`--candidate-out`, `--threshold` (overrides the threshold stored in
`<model>.threshold`), `--train-dir` + `--unseen-country-threshold` (a
separate, typically stricter, threshold for Source 1 entities whose
country never appears in `<train-dir>/train_source1.tsv` — the unseen set
is derived from the data, no country is named in code), `--gate` /
`--delta` (optional per-entity rules, off by default; gave no meaningful
validation gain in testing), `--scores-cache` (reuse scores precomputed by
the *same* model for the *same* features file, skipping inference — the
script errors out if the row count doesn't match; see "Fast iteration"
below for why this exists). With all of these at their defaults the output
is exactly "score >= threshold".

## Repo structure

```
code/business_entity_resolution/
├── src/                C++ (blocking, feature extraction, export) + Python (train/predict)
│   ├── normalize.hpp/.cpp     raw text -> compact cleaned string (abbreviation expansion)
│   ├── tokens.hpp              cleaned string -> token/trigram/bigram hashes, on demand
│   ├── record.hpp              Record struct: id, country, cleaned nameCore, cleaned addr
│   ├── tsv_io.hpp/.cpp         TSV parsing/writing (parallel cleaning on load)
│   ├── inverted_index.hpp      CSR-style token -> record-index postings list
│   ├── blocking.hpp/.cpp       per-country candidate generation
│   ├── levenshtein.hpp         bounded, length-capped edit distance
│   ├── features.hpp/.cpp       pairwise similarity feature vector (15 features)
│   ├── metrics.hpp             per-entity F_0.5
│   ├── train_main.cpp          train-fit + validation blocking/feature export (multi-threaded), self-check
│   ├── predict_main.cpp        test-set blocking/feature export (multi-threaded)
│   ├── train_model.py          LightGBM training + hyperparameter search + threshold tuning
│   ├── predict_model.py        LightGBM inference -> matching_results.tsv / candidate_pairs.tsv
│   └── make_dev_subset.py      dev-tooling only, NOT part of the graded pipeline: builds a
│                                small stratified subset for fast local iteration
├── CMakeLists.txt
└── requirements.txt     pinned Python dependencies
Documentation_template.md   formal methodology write-up (required challenge deliverable)
utils/validate_submission.py   the challenge's own submission-format validator (not written by us)
```

`dataset/`, `output/`, and `code/business_entity_resolution/{artifacts,build}/`
are generated or externally-sourced and excluded via `.gitignore`:

```
artifacts/
  train_features.tsv, val_features.tsv, val_entities.tsv   C++ export output (train_er)
  test_features.tsv                                         C++ export output (predict_er)
  model.txt, model.txt.threshold                             LightGBM model + tuned threshold (train_model.py)
```

## Measured results (full dataset)

Full run: 2,206,821 Source 1 training entities, 1,732,544 Source 1 test
entities (`--threads 8`, chosen after benchmarking — see "Threading note").

**Blocking:**
- Recall ceiling: 94.66% (train-fit split), 94.61% (validation split) —
  both well up from an earlier, simpler version of this pipeline's ~78%,
  from a wider feature/candidate set and running on the full corpus
  instead of a dev subset.
- Oracle ceiling (macro F_0.5 if the matching model were perfect, given
  current blocking) on the validation split: **0.9804** overall —
  singleton 1.0 exactly (n=12,161), single-match 0.9416 (n=11,931),
  multi-match 0.9815 (n=196,590).
- Self-check (500 randomly sampled train-fit entities, recomputed
  independently and diffed against the parallel-export output): **PASS**.

**Model (LightGBM, random hyperparameter search, 8 trials over num_leaves x
max_depth x min_child_samples x learning_rate x reg_lambda):**
- Best config: `num_leaves=255, max_depth=12, min_child_samples=50,
  learning_rate=0.08, reg_lambda=0.1`
- Tuned decision threshold: **0.875**
- **Macro F_0.5 on held-out validation split: 0.9289** (vs. 0.7309 for the
  earlier single-stage logistic-regression baseline — +0.198). At the
  untuned 0.5 threshold: 0.8711.
- That's 94.8% of the 0.9804 oracle ceiling (0.9289 / 0.9804) — the
  remaining gap at validation time was mostly model quality, not blocking
  recall.
- F_0.5 breakdown by bucket (tuned threshold): singleton 0.8618 (n=13,472,
  oracle says this should be a perfect 1.0 — every point lost here is a
  false match predicted on a true singleton), single-match 0.8392
  (n=16,149, the hardest bucket — the precision-heavy metric punishes any
  extra false match hard when there's only one true match), multi-match
  0.9412 (n=191,061).
- Feature importances (gain): `name_trigram_jaccard` dominates
  (101.4M, ~2.5x the runner-up), then `addr_token_jaccard` (41.3M),
  `addr_numeric_token_jaccard` (23.3M), `name_lev_sim` (20.4M),
  `weighted_block_score_norm` (20.0M); weakest were
  `addr_numeric_overlap_norm`, `name_first_token_match`,
  `addr_primary_num_exact`.

**Inference on the full test set (submitted configuration: France
threshold 0.96, US/India threshold 0.875):**
- 86,627,139 candidates generated (every one of the 1,732,544 test
  entities got at least one candidate), 5,370,749 matches predicted.
- 117,281 / 1,732,544 (6.77%) predicted as singletons.
- Both output files pass `utils/validate_submission.py`.

### France-specific tuning

France never appears in `train_source1.tsv`, so there is no held-out
validation signal for it at all — the 0.9289 validation macro F_0.5 above
is US/India only (0.9465 US, 0.9026 India when broken out). A post-hoc
diagnostic on the test features (comparing France's score/feature
distributions against US/India, all scored with the same model) found
France's candidate pairs score systematically higher even for likely
non-matches: mean pair score 0.109 vs. 0.085-0.086 for US/India, and 5.6%
of France pairs land in the "uncertain" 0.3-0.95 score band vs. 3.2-4.0%
for US/India. Address-similarity features specifically are elevated
(`addr_token_jaccard` 0.51 vs. ~0.35, `addr_trigram_jaccard` 0.57 vs.
~0.38-0.39, `addr_lev_sim` 0.61 vs. ~0.43-0.53) — the likely cause is
shared French street vocabulary (rue, avenue, boulevard, ...) inflating
token/trigram overlap between genuinely different addresses, since the
normalization abbreviation table (`normalize.cpp`) was authored around
US/India patterns only and was never extended for France.

Since there's no France ground truth to validate against offline, this was
tuned directly against the competition leaderboard (the only real signal
available) rather than the validation split:

| Submission | France threshold | Leaderboard macro F_0.5 |
|---|---|---|
| Uniform threshold (same 0.875 everywhere) | 0.875 | 0.870 |
| **Stricter threshold for unseen-in-training countries** | **0.96** | **0.878** |

+0.008 from this one change, with zero retraining and zero change to
blocking or candidates (`candidate_pairs.tsv` is byte-identical between the
two — only which candidates cross the match threshold changes). This is
implemented generically in `predict_model.py` (`--train-dir` +
`--unseen-country-threshold`): the "unseen" country set is derived from
whatever `--train-dir` actually contains, not hard-coded to "France", so
the same flag would apply correctly to any future test set with a country
absent from training.

**What's not fully resolved:** only two leaderboard submissions were used
to test France thresholds before the deadline (0.875 uniform → 0.870;
France-only 0.96 → 0.878). 0.96 is confirmed better than 0.875 but **not**
confirmed optimal — a middle value (0.90) was generated and
format-validated but never leaderboard-tested, and nothing above 0.96 was
tried. A proper sweep was planned but not completed due to limited
leaderboard upload budget and time. This whole line of investigation was
driven by comparing offline validation (0.9289) against the actual
leaderboard score (0.870) and inferring France was the gap, since France
has no offline signal at all; the 0.878 result confirms the *direction*
was right, but it's one data point, not an isolated France-only score — it's
possible some of the +0.008 is leaderboard-subset noise rather than purely
a France effect. Treat "0.96 is optimal for France" as a working hypothesis,
not a settled fact. See "Future work" below for how to continue this.

### Threading note

Per-entity blocking + feature computation is parallelized (`--threads`),
but scaling is sub-linear past the physical core count on the development
machine (8 physical / 16 logical cores): a controlled benchmark on an
identical 60,000-entity sample measured 243.2/169.9 entities/sec
(train-fit/validation) at the default auto-detected 14 threads vs.
**259.8/246.4 entities/sec at 8 threads** (+6.8% / +45% — the validation
split does more feature computations per entity than the negative-sampled
train-fit split, so it's more sensitive to the extra contention from
over-threading). CPU utilization was sampled at 90.6-94.6% (average ~92.5%)
during the 14-thread run, confirming the cores were genuinely busy rather
than blocked on locks or I/O (utilization wasn't separately re-sampled at 8
threads — the inference that the same holds there rests on it being the
identical workload, not a repeated measurement). The most likely
explanation for 14 threads nonetheless being slower is memory-bandwidth/
cache pressure from hyperthreading on this hash-map- and
edit-distance-heavy workload — a hardware ceiling, not a fixable bug.
`--threads 8` was used for both training and test runs as a result; the
code's own auto-detect (`hardware_concurrency() - 2`) will only pick 8 on
an 8-thread machine, so pass `--threads` explicitly to match your own
physical core count rather than relying on auto-detect on a hyperthreaded
CPU.

## Fast iteration: score-caching

Re-scoring the full 86.6M-row test set from scratch with the model took
roughly 45 minutes — too slow to test several threshold ideas against a
limited leaderboard upload budget. `predict_model.py --scores-cache` scores
every row once, saves the raw scores to a `.npy` file, and reuses that
cache for every subsequent threshold/rule experiment — roughly 15 minutes
instead of 45, since it's pure post-processing with no model inference.
Regenerate the cache (`booster.predict(...)` over the full feature file,
`np.save`) once after any retrain, then experiment with thresholds/rules
against it as many times as you like for the cost of one scoring pass.

## Future work

Ranked by expected payoff:

1. **Sweep the France threshold properly** (highest payoff, least effort).
   With a cached score file, generating and validator-checking a new
   France-threshold variant costs ~15 minutes, no retraining. Try a real
   grid (0.90, 0.93, 0.96, 0.98, 0.99) to find the actual optimum instead
   of the two-point comparison time allowed for here.
2. **Extend `normalize.cpp`'s abbreviation table for French** — it
   currently only handles US/India patterns. Adding French address
   vocabulary (rue, avenue, boulevard, cedex, arrondissement) and
   legal-entity suffixes (SARL/SAS/SA) attacks the *cause* of the France
   score inflation instead of just raising the threshold to compensate —
   should help precision without sacrificing recall the way a blunt
   threshold increase does.
3. **Close the singleton/single-match precision gap.** These buckets score
   0.86/0.84 against an achievable 1.0/0.94 (multi-match is already at
   0.94 against 0.98) — the model over-predicts matches specifically on
   entities that should have zero or one true match, which the
   precision-heavy metric punishes hard. Ideas: a feature capturing the
   *relative* gap between an entity's top and second-best candidate score
   (right now every candidate is scored independently); more targeted
   tuning of the existing `--gate`/`--delta` per-entity rules specifically
   for these buckets rather than overall macro F_0.5 (a first pass on
   validation didn't help, but wasn't targeted at this).
4. **A larger hyperparameter search.** Only 8 of 576 possible combinations
   were tried. `best_iteration` landed at 1997 of a 2000-round cap,
   suggesting it hadn't fully plateaued — more search budget or more
   boosting rounds likely has headroom left.
5. **Blocking recall** (lower priority — there's more headroom in the
   model right now than in blocking). `train_er --dump-examples <path>`
   dumps the actual false-negative blocking misses with name/address side
   by side; read a sample before guessing at a fix. Consider phonetic
   blocking keys (Soundex/Metaphone-style) for typo/transliteration
   variants, or raising `--max-candidates` beyond 50.
6. **Features considered but not built:** TF-IDF cosine similarity for
   name/address (a genuinely different signal from the Jaccard/Levenshtein
   used here — weights rare tokens more heavily); phonetic or cross-script
   name matching (e.g. Devanagari vs. Latin transliteration) — currently
   entirely unmodeled, with address similarity alone carrying those cases.
7. **A country-conditional feature or per-country calibration** (e.g.
   Platt scaling per country) would let the model express uncertainty on
   an unfamiliar country more precisely than a single extra threshold can.

