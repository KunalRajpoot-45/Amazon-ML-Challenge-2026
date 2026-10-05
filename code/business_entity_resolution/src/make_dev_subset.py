#!/usr/bin/env python3
"""Builds a stratified ~8% dev subset of the training data for fast
iteration during development. Unlike a naive head-N truncation, this keeps
recall-ceiling measurements meaningful: every Source1 entity sampled has
ALL of its ground-truth-referenced Source2/Source3 records included, plus
an ~8% random background sample of the remaining pool (so blocking still
has a realistic volume of false-candidate noise to filter).

This script and its output are dev tooling only -- not part of the graded
pipeline or the final submission.
"""
import csv
import os
import random
import sys

SRC_DIR = sys.argv[1] if len(sys.argv) > 1 else "dataset/train"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "dataset/dev"
SAMPLE_RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
SEED = 42

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(r)
        for row in r:
            if row:
                yield row


print("Pass 1: sampling Source1 entities and collecting referenced ids...")
gt_path = os.path.join(SRC_DIR, "train_ground_truth.tsv")
dev_s1_ids = set()
referenced = set()
gt_rows_for_dev = []
for row in read_rows(gt_path):
    s1id = row[0]
    matched = row[1].split(",") if len(row) > 1 and row[1] else []
    if random.random() < SAMPLE_RATE:
        dev_s1_ids.add(s1id)
        referenced.update(matched)
        gt_rows_for_dev.append((s1id, ",".join(matched)))

print(f"  dev S1 entities: {len(dev_s1_ids)}")
print(f"  referenced S2/S3 ids: {len(referenced)}")

ref_s2 = {x for x in referenced if x.startswith("S2-")}
ref_s3 = {x for x in referenced if x.startswith("S3-")}


def filter_source(fname, ref_ids):
    in_path = os.path.join(SRC_DIR, fname)
    # keep the train_*.tsv naming so --train-dir can point straight at OUT_DIR
    out_path = os.path.join(OUT_DIR, fname)
    kept = 0
    total = 0
    with open(in_path, "r", encoding="utf-8", newline="") as fin, \
         open(out_path, "w", encoding="utf-8", newline="") as fout:
        r = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        w = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
        header = next(r)
        w.writerow(header)
        for row in r:
            total += 1
            if not row:
                continue
            eid = row[0]
            if eid in ref_ids or random.random() < SAMPLE_RATE:
                w.writerow(row)
                kept += 1
    print(f"  {fname}: kept {kept}/{total}")
    return out_path


print("Pass 2: filtering source1...")
s1_path = os.path.join(SRC_DIR, "train_source1.tsv")
out1 = os.path.join(OUT_DIR, "train_source1.tsv")
kept1 = 0
with open(s1_path, "r", encoding="utf-8", newline="") as fin, \
     open(out1, "w", encoding="utf-8", newline="") as fout:
    r = csv.reader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
    w = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
    header = next(r)
    w.writerow(header)
    for row in r:
        if row and row[0] in dev_s1_ids:
            w.writerow(row)
            kept1 += 1
print(f"  dev_source1.tsv: kept {kept1}")

print("Pass 3: filtering source2/source3 (referenced + background sample)...")
filter_source("train_source2.tsv", ref_s2)
filter_source("train_source3.tsv", ref_s3)

print("Pass 4: writing dev ground truth...")
out_gt = os.path.join(OUT_DIR, "train_ground_truth.tsv")
with open(out_gt, "w", encoding="utf-8", newline="") as fout:
    w = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
    w.writerow(["source1_entity_id", "matched_entity_ids"])
    for s1id, matched in gt_rows_for_dev:
        w.writerow([s1id, matched])
print(f"  dev_ground_truth.tsv: {len(gt_rows_for_dev)} rows")

print("Done.")
