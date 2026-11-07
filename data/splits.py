"""BnSentMix dedup + near-duplicate clustering + grouped stratified 80/10/10 split.

Steps (configs/data.yaml `bnsentmix`):
1. Exact duplicates on match-normalized text are collapsed; groups whose copies
   carry conflicting labels are dropped entirely (and listed).
2. Near-duplicates (char-shingle Jaccard >= threshold, MinHash LSH + exact
   verification) are union-found into clusters.
3. Clusters are stratified by majority label (StratifiedKFold over clusters):
   fold 0 is test, fold 1 is dev, the rest train. A near-copy can never
   straddle the train/test boundary.

Writes data/splits/bnsentmix_{train,dev,test}.jsonl (committed) and
data/splits/bnsentmix_split_report.json.

Usage:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.splits
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.clean import clean_text, normalize_for_match
from data.common import load_config, repo_path, sha256_file, setup_env, write_json
from data.minhash import MinHasher, near_dup_clusters

SPLITS = ("train", "dev", "test")


@dataclass
class Row:
    """One BnSentMix row kept after dedup."""

    row_id: int
    text: str
    label: int
    norm: str
    norm_sha: str


def load_rows(csv_path: Path, text_col: str, label_col: str) -> list[tuple[int, str, int]]:
    """(row_id, text, label) for every CSV row; row_id is the 0-based data-row index."""
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return [(i, r[text_col], int(r[label_col])) for i, r in enumerate(csv.DictReader(f))]


def dedup_exact(rows: list[tuple[int, str, int]]) -> tuple[list[Row], dict[str, Any]]:
    """Collapse exact duplicates (normalized); drop groups with conflicting labels."""
    groups: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
    for row_id, text, label in rows:
        groups[normalize_for_match(text)].append((row_id, text, label))
    kept: list[Row] = []
    conflicts, collapsed, empty = [], 0, 0
    for norm, members in groups.items():
        if not norm:
            empty += len(members)
            continue
        labels = {m[2] for m in members}
        if len(labels) > 1:
            conflicts.append({"rows": [m[0] for m in members], "labels": sorted(labels), "text": members[0][1]})
            continue
        collapsed += len(members) - 1
        row_id, text, label = min(members)
        kept.append(Row(row_id, text, label, norm, hashlib.sha256(norm.encode("utf-8")).hexdigest()))
    kept.sort(key=lambda r: r.row_id)
    stats = {
        "input_rows": len(rows),
        "duplicate_groups": sum(1 for m in groups.values() if len(m) > 1),
        "rows_collapsed": collapsed,
        "conflicting_label_groups": len(conflicts),
        "rows_dropped_conflicting": sum(len(c["rows"]) for c in conflicts),
        "rows_dropped_empty": empty,
        "kept_rows": len(kept),
        "conflicts": conflicts,
    }
    return kept, stats


def assign_splits(rows: list[Row], clusters: list[int], n_folds: int, seed: int) -> dict[str, list[Row]]:
    """Grouped stratified split: fold 0 → test, fold 1 → dev, rest → train.

    Clusters (not rows) are stratified by their majority label with StratifiedKFold,
    then expanded back to rows, so every cluster lands in exactly one split.
    (sklearn's StratifiedGroupKFold was tried first; its greedy balancing drifted
    up to 9 points per class on synthetic data, so it is not used.)
    """
    from sklearn.model_selection import StratifiedKFold

    members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        members[c].append(i)
    cluster_ids = sorted(members)
    cluster_labels = [Counter(rows[i].label for i in members[c]).most_common(1)[0][0] for c in cluster_ids]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in skf.split(cluster_ids, cluster_labels)]
    test_c = {cluster_ids[j] for j in folds[0]}
    dev_c = {cluster_ids[j] for j in folds[1]}
    out: dict[str, list[Row]] = {s: [] for s in SPLITS}
    for i, r in enumerate(rows):
        c = clusters[i]
        out["test" if c in test_c else "dev" if c in dev_c else "train"].append(r)
    return out


def build_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run the full split pipeline and write manifests; returns the split report."""
    bcfg = cfg["bnsentmix"]
    csv_path = repo_path(cfg["paths"]["raw_dir"]) / "bnsentmix" / "dataset.csv"
    rows = load_rows(csv_path, bcfg["text_column"], bcfg["label_column"])
    kept, dedup_stats = dedup_exact(rows)

    hasher = MinHasher.from_config(cfg)
    k = int(cfg["minhash"]["shingle_chars"])
    clusters = near_dup_clusters([r.norm for r in kept], k, hasher, float(bcfg["near_dup_jaccard"]))
    sizes = Counter(clusters)
    splits = assign_splits(kept, clusters, int(bcfg["split"]["n_folds"]), int(bcfg["split"]["seed"]))

    # Invariant: no near-duplicate cluster spans two splits.
    cluster_split: dict[int, str] = {}
    idx_of = {r.row_id: i for i, r in enumerate(kept)}
    for name, members in splits.items():
        for r in members:
            c = clusters[idx_of[r.row_id]]
            if cluster_split.setdefault(c, name) != name:
                raise AssertionError(f"cluster {c} spans {cluster_split[c]} and {name}")

    out_dir = repo_path(cfg["paths"]["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {int(k_): v for k_, v in bcfg["label_names"].items()}
    for name, members in splits.items():
        with (out_dir / f"bnsentmix_{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in members:
                f.write(json.dumps({"row_id": r.row_id, "label": r.label, "norm_sha256": r.norm_sha}) + "\n")

    report = {
        "source_csv_sha256": sha256_file(csv_path),
        "source_revision": cfg["sources"]["bnsentmix"]["revision"],
        "dedup": dedup_stats,
        "near_dup": {
            "jaccard": bcfg["near_dup_jaccard"],
            "clusters": len(sizes),
            "multi_member_clusters": sum(1 for s in sizes.values() if s > 1),
            "rows_in_multi_member_clusters": sum(s for s in sizes.values() if s > 1),
            "largest_cluster": max(sizes.values()),
        },
        "splits": {
            name: {
                "rows": len(m),
                "share": round(len(m) / len(kept), 4),
                "labels": {names[lab]: c for lab, c in sorted(Counter(r.label for r in m).items())},
            }
            for name, m in splits.items()
        },
    }
    write_json(out_dir / "bnsentmix_split_report.json", report)
    return report


def load_split_rows(cfg: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Rows of one split joined back to the CSV: row_id, label, text (raw), clean, norm."""
    csv_path = repo_path(cfg["paths"]["raw_dir"]) / "bnsentmix" / "dataset.csv"
    bcfg = cfg["bnsentmix"]
    by_id = {i: (t, lab) for i, t, lab in load_rows(csv_path, bcfg["text_column"], bcfg["label_column"])}
    out = []
    path = repo_path(cfg["paths"]["splits_dir"]) / f"bnsentmix_{split}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        text, label = by_id[rec["row_id"]]
        if label != rec["label"]:
            raise ValueError(f"label mismatch for row {rec['row_id']}")
        norm = normalize_for_match(text)
        if hashlib.sha256(norm.encode("utf-8")).hexdigest() != rec["norm_sha256"]:
            raise ValueError(f"text hash mismatch for row {rec['row_id']}: CSV changed?")
        out.append({"row_id": rec["row_id"], "label": label, "text": text, "clean": clean_text(text), "norm": norm})
    return out


if __name__ == "__main__":
    config = load_config()
    setup_env(config)
    rep = build_splits(config)
    print(json.dumps({k: v for k, v in rep.items() if k != "dedup"} | {"dedup": {k: v for k, v in rep["dedup"].items() if k != "conflicts"}}, indent=2))
