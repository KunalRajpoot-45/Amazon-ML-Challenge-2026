#!/usr/bin/env python3
"""Trains a LightGBM binary classifier on the candidate-pair features
written by train_er (the C++ blocking/feature-extraction stage), tunes the
decision threshold to maximize macro per-entity F_0.5 on the held-out
validation split (matching the competition's actual metric, not AUC/F1),
and saves the model in LightGBM's native text format (portable, not a
pickle) plus a small sidecar file with the tuned threshold.

Hyperparameters (num_leaves, max_depth, min_data_in_leaf, learning_rate,
n_estimators) are chosen by a random search: each candidate config is
trained with early stopping on the validation set, and configs are ranked
by the *actual* per-entity macro F_0.5 metric (at that config's own
best threshold), not by LightGBM's internal AUC/logloss. See
--search-iterations / --search-space below.

This model is trained from scratch on the provided training data only --
not a pretrained model, so it carries none of the license/parameter-count
constraints that apply to pretrained models. LightGBM itself (the training
algorithm/library) is MIT-licensed.

Usage:
    python3 code/business_entity_resolution/src/train_model.py \\
        --train-features code/business_entity_resolution/artifacts/train_features.tsv \\
        --val-features code/business_entity_resolution/artifacts/val_features.tsv \\
        --val-entities code/business_entity_resolution/artifacts/val_entities.tsv \\
        --model-out code/business_entity_resolution/artifacts/model.txt
"""
import argparse
import itertools
import random
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COLS = ["source1_entity_id", "candidate_entity_id"]
LABEL_COL = "label"


def feature_columns(path):
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    return [c for c in header if c not in ID_COLS and c != LABEL_COL]


def load_xy(path, feat_cols, with_id=False):
    usecols = feat_cols + [LABEL_COL] + (["source1_entity_id"] if with_id else [])
    dtype = {c: np.float32 for c in feat_cols}
    dtype[LABEL_COL] = np.int8
    df = pd.read_csv(path, sep="\t", usecols=usecols, dtype=dtype)
    return df


def f0_5_from_counts(tp, pred_count, truth_count):
    if pred_count == 0 and truth_count == 0:
        return 1.0
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / truth_count if truth_count else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def macro_f0_5_at_threshold(scores, labels, group_codes, num_groups, truth_count_by_group,
                             zero_cand_f0_5_sum, num_zero_cand, threshold):
    pred = (scores >= threshold)
    tp = np.bincount(group_codes, weights=(pred & (labels == 1)).astype(np.float64), minlength=num_groups)
    pred_count = np.bincount(group_codes, weights=pred.astype(np.float64), minlength=num_groups)

    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, truth_count_by_group, out=np.zeros_like(tp), where=truth_count_by_group > 0)
    denom = 0.25 * precision + recall
    f = np.divide(1.25 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    # predCount==0 & truthCount==0 -> singleton correctly predicted empty -> F=1
    singleton_mask = (pred_count == 0) & (truth_count_by_group == 0)
    f[singleton_mask] = 1.0

    total = f.sum() + zero_cand_f0_5_sum
    n = num_groups + num_zero_cand
    return total / n if n else 0.0


SEARCH_SPACE = {
    "num_leaves": [31, 63, 127, 255],
    "max_depth": [-1, 8, 12],
    "min_child_samples": [10, 20, 50, 100],
    "learning_rate": [0.03, 0.05, 0.08, 0.12],
    "reg_lambda": [0.1, 1.0, 5.0],
}


def best_threshold(scores, y_val, entity_codes, num_groups, truth_count_by_group,
                    zero_cand_f0_5_sum, num_zero_cand):
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.005, 1.0, 0.005):
        f = macro_f0_5_at_threshold(scores, y_val, entity_codes, num_groups, truth_count_by_group,
                                     zero_cand_f0_5_sum, num_zero_cand, t)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--val-features", required=True)
    ap.add_argument("--val-entities", required=True)
    ap.add_argument("--model-out", required=True)
    ap.add_argument("--search-iterations", type=int, default=12,
                     help="number of random hyperparameter configs to try (0 = skip search, "
                          "use --num-leaves/--learning-rate/etc. directly)")
    ap.add_argument("--search-boost-round", type=int, default=300,
                     help="num_boost_round cap used for search trials (kept small so the "
                          "search finishes in reasonable time; the winning config is then "
                          "retrained with --num-boost-round for the final model)")
    ap.add_argument("--search-early-stopping-rounds", type=int, default=30)
    ap.add_argument("--search-seed", type=int, default=42)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-boost-round", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--min-child-samples", type=int, default=20)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:.1f}s] {msg}", flush=True)

    feat_cols = feature_columns(args.train_features)
    log(f"feature columns ({len(feat_cols)}): {feat_cols}")

    log("loading train features...")
    train_df = load_xy(args.train_features, feat_cols, with_id=False)
    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df[LABEL_COL].to_numpy(dtype=np.int8)
    del train_df
    log(f"train: {len(y_train)} rows, {y_train.sum()} positive")

    log("loading validation features...")
    val_df = load_xy(args.val_features, feat_cols, with_id=True)
    X_val = val_df[feat_cols].to_numpy(dtype=np.float32)
    y_val = val_df[LABEL_COL].to_numpy(dtype=np.int8)
    val_entity_ids = val_df["source1_entity_id"].to_numpy()
    del val_df
    log(f"val: {len(y_val)} rows, {y_val.sum()} positive")

    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = float(np.clip(neg / max(pos, 1), 0.5, 10.0))
    log(f"scale_pos_weight={scale_pos_weight:.4f}")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=feat_cols, free_raw_data=False)

    entity_codes, uniques = pd.factorize(val_entity_ids)
    num_groups = len(uniques)
    truth_count_by_group = np.bincount(entity_codes, weights=y_val.astype(np.float64), minlength=num_groups)

    ent_df = pd.read_csv(args.val_entities, sep="\t")
    zero_cand = ent_df[ent_df["has_candidates"] == 0]
    zero_cand_f0_5_sum = float((zero_cand["truth_count"] == 0).sum())  # singleton -> F=1, else F=0
    num_zero_cand = len(zero_cand)
    log(f"validation entities: {num_groups} with candidates, {num_zero_cand} with zero candidates")

    def base_params(cfg, seed):
        return {
            "objective": "binary",
            "metric": ["auc", "binary_logloss"],
            "boosting_type": "gbdt",
            # The same lgb.Dataset is reused across search trials with different
            # min_child_samples values; feature_pre_filter (on by default) locks
            # in filtering decisions from whichever value hits the Dataset first
            # and throws a fatal error if a later trial uses a smaller one.
            "feature_pre_filter": False,
            "num_leaves": cfg["num_leaves"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
            "min_child_samples": cfg["min_child_samples"],
            "reg_lambda": cfg["reg_lambda"],
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "scale_pos_weight": scale_pos_weight,
            "seed": seed,
            "verbose": -1,
        }

    def train_and_score(cfg, num_boost_round, early_stopping_rounds, seed, log_eval_period=0):
        params = base_params(cfg, seed)
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        if log_eval_period:
            callbacks.append(lgb.log_evaluation(log_eval_period))
        booster = lgb.train(
            params, train_set, num_boost_round=num_boost_round,
            valid_sets=[valid_set], valid_names=["val"], callbacks=callbacks,
        )
        scores = booster.predict(X_val, num_iteration=booster.best_iteration)
        t, f = best_threshold(scores, y_val, entity_codes, num_groups, truth_count_by_group,
                               zero_cand_f0_5_sum, num_zero_cand)
        return booster, t, f

    if args.search_iterations > 0:
        keys = list(SEARCH_SPACE.keys())
        all_combos = list(itertools.product(*(SEARCH_SPACE[k] for k in keys)))
        rng = random.Random(args.search_seed)
        n_trials = min(args.search_iterations, len(all_combos))
        trial_combos = rng.sample(all_combos, n_trials)
        log(f"hyperparameter random search: {n_trials} trials out of {len(all_combos)} possible "
            f"combos (search_boost_round={args.search_boost_round})")

        best_cfg, best_f, best_t = None, -1.0, 0.5
        results = []
        for i, combo in enumerate(trial_combos):
            cfg = dict(zip(keys, combo))
            trial_t0 = time.time()
            _, t, f = train_and_score(cfg, args.search_boost_round, args.search_early_stopping_rounds,
                                       args.search_seed)
            dt = time.time() - trial_t0
            results.append((f, t, cfg))
            log(f"trial {i + 1}/{n_trials} ({dt:.1f}s): {cfg} -> macro F0.5={f:.6f} (t={t:.3f})")
            if f > best_f:
                best_f, best_t, best_cfg = f, t, cfg

        results.sort(key=lambda r: -r[0])
        print("\n=== Hyperparameter search results (top 5) ===")
        for f, t, cfg in results[:5]:
            print(f"  F0.5={f:.6f} t={t:.3f} {cfg}")
        log(f"best search config: {best_cfg} -> macro F0.5={best_f:.6f} (search_boost_round trial)")

        log(f"retraining best config at full num_boost_round={args.num_boost_round} for the final model...")
        booster, best_t, best_f = train_and_score(best_cfg, args.num_boost_round,
                                                   args.early_stopping_rounds, args.seed, log_eval_period=50)
        log(f"final model: best_iteration={booster.best_iteration}, macro F0.5(val)={best_f:.6f}, "
            f"threshold={best_t:.4f}")
    else:
        best_cfg = {
            "num_leaves": args.num_leaves, "max_depth": args.max_depth,
            "learning_rate": args.learning_rate, "min_child_samples": args.min_child_samples,
            "reg_lambda": args.reg_lambda,
        }
        log(f"search disabled (--search-iterations 0); training fixed config: {best_cfg}")
        booster, best_t, best_f = train_and_score(best_cfg, args.num_boost_round,
                                                   args.early_stopping_rounds, args.seed, log_eval_period=50)
        log(f"training done, best_iteration={booster.best_iteration}")

    scores_at_final = booster.predict(X_val, num_iteration=booster.best_iteration)
    f_at_05 = macro_f0_5_at_threshold(scores_at_final, y_val, entity_codes, num_groups, truth_count_by_group,
                                       zero_cand_f0_5_sum, num_zero_cand, 0.5)
    log(f"macro F0.5(val) @0.5 = {f_at_05:.6f}  (tuned threshold {best_t:.4f} -> {best_f:.6f})")

    print("\n=== Comparison ===")
    print(f"  Old logistic regression baseline: F0.5 = 0.7309")
    print(f"  LightGBM (this run, tuned):        F0.5 = {best_f:.4f}")
    print(f"  LightGBM (this run, @0.5):         F0.5 = {f_at_05:.4f}")
    print(f"  best hyperparameters:               {best_cfg}")

    booster.save_model(args.model_out, num_iteration=booster.best_iteration)
    with open(args.model_out + ".threshold", "w") as f:
        f.write(str(best_t))
    log(f"saved model to {args.model_out} (+ .threshold sidecar)")

    importances = booster.feature_importance(importance_type="gain")
    order = np.argsort(-importances)
    print("\nfeature importances (gain):")
    for idx in order:
        print(f"  {feat_cols[idx]:<28} {importances[idx]:.1f}")

    # Per-entity F0.5 breakdown by bucket (singleton / single-match / multi-match),
    # at the tuned threshold, so gaps to a very high macro F0.5 target are visible.
    print("\n=== F0.5 breakdown by bucket (tuned threshold) ===")
    pred = scores_at_final >= best_t
    tp = np.bincount(entity_codes, weights=(pred & (y_val == 1)).astype(np.float64), minlength=num_groups)
    pred_count = np.bincount(entity_codes, weights=pred.astype(np.float64), minlength=num_groups)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, truth_count_by_group, out=np.zeros_like(tp), where=truth_count_by_group > 0)
    denom = 0.25 * precision + recall
    f_per_entity = np.divide(1.25 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    singleton_mask = (pred_count == 0) & (truth_count_by_group == 0)
    f_per_entity[singleton_mask] = 1.0

    bucket = np.where(truth_count_by_group == 0, 0, np.where(truth_count_by_group == 1, 1, 2))
    bucket_names = {0: "singleton", 1: "single-match", 2: "multi-match"}
    for b in (0, 1, 2):
        mask = bucket == b
        n = int(mask.sum()) + (num_zero_cand if b == 0 else 0)
        if b == 0:
            f_sum = f_per_entity[mask].sum() + zero_cand_f0_5_sum
        else:
            f_sum = f_per_entity[mask].sum()
        print(f"  {bucket_names[b]:<14} n={n:<10} macro F0.5={(f_sum / n) if n else 0.0:.6f}")


if __name__ == "__main__":
    main()
