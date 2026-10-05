#!/usr/bin/env python3
"""Loads the LightGBM model trained by train_model.py and its tuned
threshold, scores every test candidate pair written by predict_er (the C++
blocking/feature-extraction stage), and writes the two required submission
files: output/matching_results.tsv and output/candidate_pairs.tsv.

Streams the (potentially tens-of-millions-of-rows) test feature file in
chunks rather than loading it whole, to keep memory bounded; per-entity
candidate rows are contiguous in that file (predict_er writes them in
source1 iteration order), so chunk boundaries are handled by carrying over
a possibly-incomplete trailing entity group to the next chunk.

Decision rule for a candidate of Source 1 entity e:
    score >= thr(e)  AND  top(e) >= --gate  AND  score >= top(e) - --delta
where top(e) is e's highest candidate score and thr(e) is the tuned threshold,
or --unseen-country-threshold if e's country never appears in the training
Source 1 data (--train-dir). With the defaults (gate 0, delta 1, no unseen
threshold) this is exactly "score >= threshold".

Usage:
    python3 code/business_entity_resolution/src/predict_model.py \\
        --test-dir dataset/test \\
        --features code/business_entity_resolution/artifacts/test_features.tsv \\
        --model code/business_entity_resolution/artifacts/model.txt \\
        --matching-out output/matching_results.tsv \\
        --candidate-out output/candidate_pairs.tsv
"""
import argparse
import csv
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ID_COLS = ["source1_entity_id", "candidate_entity_id"]
CHUNK_ROWS = 2_000_000


def feature_columns(path):
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    return [c for c in header if c not in ID_COLS]


def read_ids_and_countries(path):
    ids, countries = [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(r)
        ci = header.index("country")
        for row in r:
            if row:
                ids.append(row[0])
                countries.append(row[ci] if ci < len(row) else "")
    return ids, countries


def process_group(df, gate, delta, candidates_by_entity, matches_by_entity):
    df = df.sort_values(ID_COLS, kind="stable")
    top = df.groupby("source1_entity_id", sort=False)["_score"].transform("max")
    pred = (df["_score"] >= df["_thr"]) & (top >= gate) & (df["_score"] >= top - delta)
    cands = df.groupby("source1_entity_id", sort=False)["candidate_entity_id"].agg(",".join)
    matches = df[pred].groupby("source1_entity_id", sort=False)["candidate_entity_id"].agg(",".join)
    candidates_by_entity.update(cands.to_dict())
    matches_by_entity.update(matches.to_dict())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--matching-out", required=True)
    ap.add_argument("--candidate-out", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                     help="override the threshold stored in <model>.threshold")
    ap.add_argument("--train-dir", default=None,
                     help="training data dir; its train_source1.tsv defines which countries were seen in training")
    ap.add_argument("--unseen-country-threshold", type=float, default=None,
                     help="threshold for entities whose country is absent from --train-dir (default: same as --threshold)")
    ap.add_argument("--gate", type=float, default=0.0,
                     help="predict no matches for an entity unless its top candidate score is >= gate")
    ap.add_argument("--delta", type=float, default=1.0,
                     help="only match candidates scoring within delta of the entity's top score")
    ap.add_argument("--scores-cache", default=None,
                     help="precomputed scores (.npy, one per --features row, same order, from the SAME --model); "
                          "skips model inference")
    args = ap.parse_args()

    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:.1f}s] {msg}", flush=True)

    if args.threshold is not None:
        threshold = args.threshold
    else:
        with open(args.model + ".threshold") as f:
            threshold = float(f.read().strip())
    log(f"threshold={threshold} gate={args.gate} delta={args.delta} "
        f"unseen_country_threshold={args.unseen_country_threshold}")

    all_ids, all_countries = read_ids_and_countries(os.path.join(args.test_dir, "test_source1.tsv"))
    log(f"test_source1.tsv: {len(all_ids)} entities")

    unseen_thr = {}
    if args.unseen_country_threshold is not None:
        if args.train_dir is None:
            ap.error("--unseen-country-threshold requires --train-dir")
        _, train_countries = read_ids_and_countries(os.path.join(args.train_dir, "train_source1.tsv"))
        seen = set(train_countries)
        unseen = sorted(set(all_countries) - seen)
        unseen_thr = {e: args.unseen_country_threshold for e, c in zip(all_ids, all_countries) if c not in seen}
        log(f"countries seen in training: {sorted(seen)}; unseen in test: {unseen} "
            f"({len(unseen_thr)} entities get threshold {args.unseen_country_threshold})")

    feat_cols = feature_columns(args.features)
    if args.scores_cache:
        cache = np.load(args.scores_cache, mmap_mode="r")
        booster = None
        reader = pd.read_csv(args.features, sep="\t", usecols=ID_COLS, dtype=str, chunksize=CHUNK_ROWS)
        log(f"using cached scores ({len(cache)} rows) from {args.scores_cache}")
    else:
        booster = lgb.Booster(model_file=args.model)
        reader = pd.read_csv(args.features, sep="\t", dtype={c: np.float32 for c in feat_cols},
                             chunksize=CHUNK_ROWS)
        log(f"loaded model, {len(feat_cols)} features")

    candidates_by_entity = {}
    matches_by_entity = {}
    pending = None
    offset = 0
    total_rows = 0
    chunk_num = 0

    for raw in reader:
        chunk_num += 1
        if booster is None:
            raw["_score"] = cache[offset:offset + len(raw)]
        else:
            raw["_score"] = booster.predict(raw[feat_cols].to_numpy(dtype=np.float32))
        offset += len(raw)
        raw = raw[ID_COLS + ["_score"]]
        chunk = raw if pending is None else pd.concat([pending, raw], ignore_index=True)

        ids = chunk["source1_entity_id"].to_numpy()
        boundary = int(np.argmax(ids == ids[-1]))  # first index of the trailing run of the last entity

        complete_part = chunk.iloc[:boundary].copy()
        pending = chunk.iloc[boundary:].reset_index(drop=True)

        if len(complete_part) > 0:
            complete_part["_thr"] = complete_part["source1_entity_id"].map(unseen_thr).fillna(threshold)
            process_group(complete_part, args.gate, args.delta, candidates_by_entity, matches_by_entity)
        total_rows += len(complete_part)
        log(f"chunk {chunk_num}: processed {total_rows} rows, {len(candidates_by_entity)} entities so far")

    if pending is not None and len(pending) > 0:
        pending["_thr"] = pending["source1_entity_id"].map(unseen_thr).fillna(threshold)
        process_group(pending, args.gate, args.delta, candidates_by_entity, matches_by_entity)
        total_rows += len(pending)

    if booster is None and offset != len(cache):
        raise SystemExit(f"--scores-cache has {len(cache)} rows but --features has {offset}; wrong cache for this file")

    log(f"scored {total_rows} candidate pairs across {len(candidates_by_entity)} entities with candidates")

    os.makedirs(os.path.dirname(args.matching_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.candidate_out) or ".", exist_ok=True)

    n_singletons = 0
    n_total_matches = 0
    n_total_candidates = 0
    with open(args.candidate_out, "w", encoding="utf-8", newline="") as cf, \
         open(args.matching_out, "w", encoding="utf-8", newline="") as mf:
        cw = csv.writer(cf, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")
        mw = csv.writer(mf, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        mw.writerow(["source1_entity_id", "matched_entity_ids"])
        for eid in all_ids:
            cands = candidates_by_entity.get(eid, "")
            matches = matches_by_entity.get(eid, "")
            cw.writerow([eid, cands])
            mw.writerow([eid, matches])
            if matches == "":
                n_singletons += 1
            else:
                n_total_matches += matches.count(",") + 1
            if cands != "":
                n_total_candidates += cands.count(",") + 1

    log(f"wrote {args.candidate_out} and {args.matching_out}")
    print("\n=== Summary ===")
    print(f"  total Source 1 test entities: {len(all_ids)}")
    print(f"  total candidate pairs: {n_total_candidates}")
    print(f"  total predicted matches: {n_total_matches}")
    print(f"  predicted singletons: {n_singletons} ({100.0 * n_singletons / len(all_ids):.2f}%)")


if __name__ == "__main__":
    main()
