#!/usr/bin/env python3
"""Validates matching_results.tsv and candidate_pairs.tsv against the
Business Entity Resolution Challenge output-format rules, using only the
standard library. Run from the student_resource/ (project root) directory:

    python3 utils/validate_submission.py \\
        --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv \\
        --test-dir dataset/test

Prints PASS (exit 0) when both files are safe to submit, or a numbered list
of issues (exit 1). This script only reads the output files and the test
source files -- it does not compute a score.
"""
import argparse
import csv
import os
import sys

MAX_EXAMPLES = 8


def read_tsv_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        for row in reader:
            yield row
        return header


def load_test_ids(test_dir):
    def ids_from(fname, prefix):
        path = os.path.join(test_dir, fname)
        ids = set()
        if not os.path.isfile(path):
            return ids, [f"test file not found: {path}"]
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                eid = row[0]
                if not eid.startswith(prefix):
                    continue
                ids.add(eid)
        return ids, []

    s1_ids, err1 = ids_from("test_source1.tsv", "S1-")
    s2_ids, err2 = ids_from("test_source2.tsv", "S2-")
    s3_ids, err3 = ids_from("test_source3.tsv", "S3-")
    return s1_ids, s2_ids, s3_ids, err1 + err2 + err3


def parse_id_list_file(path, expected_header):
    """Returns (header_ok, rows: dict[s1id -> list[str]], duplicate_s1_rows: list,
    malformed_rows: list) where rows preserves the first occurrence only."""
    if not os.path.isfile(path):
        return False, {}, [], [f"file not found: {path}"]

    rows = {}
    duplicate_s1_rows = []
    malformed = []
    header_ok = False

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader, None)
        if header is not None and len(header) >= 2 and header[0] == "source1_entity_id" \
                and header[1] == expected_header:
            header_ok = True

        for i, row in enumerate(reader, start=2):
            if not row or row == [""]:
                continue
            if len(row) == 1:
                s1id, idlist = row[0], ""
            elif len(row) == 2:
                s1id, idlist = row[0], row[1]
            else:
                malformed.append((i, row))
                continue

            ids = [x for x in idlist.split(",")] if idlist != "" else []
            if s1id in rows:
                duplicate_s1_rows.append(s1id)
                continue
            rows[s1id] = ids

    return header_ok, rows, duplicate_s1_rows, malformed


def validate(matching_path, candidate_path, test_dir):
    issues = []
    warnings = []

    test_s1_ids, test_s2_ids, test_s3_ids, load_errs = load_test_ids(test_dir)
    for e in load_errs:
        issues.append(e)
    if load_errs:
        return issues, warnings  # can't validate further without test data

    def check_file(path, label, expected_header):
        header_ok, rows, dup_rows, malformed = parse_id_list_file(path, expected_header)
        if not os.path.isfile(path):
            issues.append(f"[{label}] file not found: {path}")
            return None
        if not header_ok:
            issues.append(f"[{label}] header must be exactly "
                           f"'source1_entity_id\\t{expected_header}'")
        if malformed:
            examples = malformed[:MAX_EXAMPLES]
            issues.append(f"[{label}] {len(malformed)} row(s) with wrong column count, "
                           f"e.g. line(s) {[m[0] for m in examples]}")
        if dup_rows:
            examples = dup_rows[:MAX_EXAMPLES]
            issues.append(f"[{label}] {len(dup_rows)} duplicate source1_entity_id row(s), "
                           f"e.g. {examples}")

        missing = test_s1_ids - rows.keys()
        if missing:
            examples = list(sorted(missing))[:MAX_EXAMPLES]
            issues.append(f"[{label}] {len(missing)} test source1 entities missing from file, "
                           f"e.g. {examples}")

        extra = rows.keys() - test_s1_ids
        if extra:
            examples = list(sorted(extra))[:MAX_EXAMPLES]
            issues.append(f"[{label}] {len(extra)} source1_entity_id value(s) not present in "
                           f"test_source1.tsv, e.g. {examples}")

        bad_ref_count = 0
        bad_ref_examples = []
        dup_within_row_count = 0
        dup_within_row_examples = []
        self_match_count = 0
        self_match_examples = []

        for s1id, ids in rows.items():
            seen = set()
            for eid in ids:
                if eid in seen:
                    dup_within_row_count += 1
                    if len(dup_within_row_examples) < MAX_EXAMPLES:
                        dup_within_row_examples.append((s1id, eid))
                seen.add(eid)

                if eid.startswith("S1-"):
                    self_match_count += 1
                    if len(self_match_examples) < MAX_EXAMPLES:
                        self_match_examples.append((s1id, eid))
                elif eid.startswith("S2-"):
                    if eid not in test_s2_ids:
                        bad_ref_count += 1
                        if len(bad_ref_examples) < MAX_EXAMPLES:
                            bad_ref_examples.append((s1id, eid))
                elif eid.startswith("S3-"):
                    if eid not in test_s3_ids:
                        bad_ref_count += 1
                        if len(bad_ref_examples) < MAX_EXAMPLES:
                            bad_ref_examples.append((s1id, eid))
                else:
                    bad_ref_count += 1
                    if len(bad_ref_examples) < MAX_EXAMPLES:
                        bad_ref_examples.append((s1id, eid))

        if self_match_count:
            issues.append(f"[{label}] {self_match_count} self-match(es) to a Source 1 id, "
                           f"e.g. {self_match_examples}")
        if bad_ref_count:
            issues.append(f"[{label}] {bad_ref_count} id(s) referencing an entity not in the "
                           f"test set (or not S2-/S3-), e.g. {bad_ref_examples}")
        if dup_within_row_count:
            issues.append(f"[{label}] {dup_within_row_count} duplicate id(s) within a single "
                           f"row's list, e.g. {dup_within_row_examples}")

        return rows

    matching_rows = check_file(matching_path, "matching_results.tsv", "matched_entity_ids")
    candidate_rows = check_file(candidate_path, "candidate_pairs.tsv", "candidate_entity_ids")

    if matching_rows is not None and candidate_rows is not None:
        not_a_candidate = 0
        examples = []
        for s1id, matched in matching_rows.items():
            cand = set(candidate_rows.get(s1id, []))
            for eid in matched:
                if eid not in cand:
                    not_a_candidate += 1
                    if len(examples) < MAX_EXAMPLES:
                        examples.append((s1id, eid))
        if not_a_candidate:
            warnings.append(f"{not_a_candidate} matched id(s) never appeared as a candidate for "
                             f"that source1 entity -- possible pipeline bug, e.g. {examples}")

    return issues, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matching", required=True, help="path to matching_results.tsv")
    ap.add_argument("--candidate", required=True, help="path to candidate_pairs.tsv")
    ap.add_argument("--test-dir", required=True, help="path to dataset/test")
    args = ap.parse_args()

    issues, warnings = validate(args.matching, args.candidate, args.test_dir)

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
        print()

    if issues:
        print(f"FAIL: {len(issues)} issue(s) found:")
        for i, issue in enumerate(issues, start=1):
            print(f"  {i}. {issue}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
