from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

# Permit both `python src/run_pipeline.py` and module-style execution.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from blocking import candidate_map, prepare  # noqa: E402
from features import make_pairs  # noqa: E402
from text import normalize_address, normalize_name  # noqa: E402


def read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).fillna("")


def normalized(df: pd.DataFrame) -> pd.DataFrame:
    required = {"entity_id", "business_name", "business_address", "country"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; found {list(df.columns)}")
    out = df.copy()
    out["business_name"] = out["business_name"].map(normalize_name)
    out["business_address"] = out["business_address"].map(normalize_address)
    return prepare(out)


def truth_sets(truth: pd.DataFrame) -> dict[str, set[str]]:
    result = {}
    for _, row in truth.iterrows():
        value = str(row.get("matched_entity_ids", ""))
        result[str(row["source1_entity_id"])] = {x for x in value.split(",") if x}
    return result


def f05(pred: set[str], actual: set[str]) -> float:
    if not pred and not actual:
        return 1.0
    if not pred or not actual:
        return 0.0
    tp = len(pred & actual)
    precision, recall = tp / len(pred), tp / len(actual)
    if precision + recall == 0:
        return 0.0
    return 1.25 * precision * recall / (0.25 * precision + recall)


def score_threshold(probs: np.ndarray, keys: list[tuple[str, str]], actual: dict[str, set[str]], threshold: float) -> float:
    predicted: dict[str, set[str]] = {}
    for p, (sid, rid) in zip(probs, keys):
        if p >= threshold:
            predicted.setdefault(sid, set()).add(rid)
    return float(np.mean([f05(predicted.get(sid, set()), actual.get(sid, set())) for sid in actual]))


def labels(keys: list[tuple[str, str]], actual: dict[str, set[str]]) -> np.ndarray:
    return np.array([int(rid in actual.get(sid, set())) for sid, rid in keys], dtype=np.int8)


def fit_model(X: pd.DataFrame, y: np.ndarray) -> RandomForestClassifier:
    if len(X) == 0:
        raise ValueError("Blocking produced no training pairs. Inspect the input data or increase blocking limits.")
    if len(np.unique(y)) < 2:
        raise ValueError("Training candidates contain only one class; candidate generation must include positives and negatives.")
    model = RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=2,
                                   class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()
    root, outdir = Path(args.data_root), Path(args.output_dir)
    train, test = root / "dataset" / "train", root / "dataset" / "test"

    s1 = normalized(read(train / "train_source1.tsv"))
    s2 = normalized(read(train / "train_source2.tsv"))
    s3 = normalized(read(train / "train_source3.tsv"))
    truth = truth_sets(read(train / "train_ground_truth.tsv"))
    t1 = normalized(read(test / "test_source1.tsv"))
    t2 = normalized(read(test / "test_source2.tsv"))
    t3 = normalized(read(test / "test_source3.tsv"))

    # Source 2 and Source 3 are distinct records but share the same feature schema.
    train_right = pd.concat([s2, s3], ignore_index=True)
    test_right = pd.concat([t2, t3], ignore_index=True)
    train_cmap = candidate_map(s1, train_right, forced=truth)
    X_all, keys_all = make_pairs(s1, train_right, train_cmap)
    y_all = labels(keys_all, truth)

    # Split by Source 1 entity to prevent pair-level leakage.
    groups = np.array([x[0] for x in keys_all])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(splitter.split(X_all, y_all, groups=groups))
    model = fit_model(X_all.iloc[train_idx], y_all[train_idx])
    val_prob = model.predict_proba(X_all.iloc[val_idx])[:, 1]
    val_keys = [keys_all[i] for i in val_idx]
    val_ids = {keys_all[i][0] for i in val_idx}
    val_truth = {k: truth.get(k, set()) for k in val_ids}
    thresholds = np.arange(0.30, 0.951, 0.025)
    scores = [(float(score_threshold(val_prob, val_keys, val_truth, t)), float(t)) for t in thresholds]
    # Prefer higher threshold on ties because the metric is precision-heavy.
    best_score, best_threshold = max(scores, key=lambda x: (x[0], x[1]))

    # Refit on every labeled candidate after selecting the threshold.
    final_model = fit_model(X_all, y_all)
    test_cmap = candidate_map(t1, test_right)
    X_test, test_keys = make_pairs(t1, test_right, test_cmap)
    test_prob = final_model.predict_proba(X_test)[:, 1] if len(X_test) else np.array([])
    predictions: dict[str, list[str]] = {sid: [] for sid in t1["entity_id"]}
    for probability, (sid, rid) in zip(test_prob, test_keys):
        if probability >= best_threshold:
            predictions[sid].append(rid)
    for sid in predictions:
        predictions[sid] = sorted(set(predictions[sid]))

    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source1_entity_id": list(predictions),
                  "matched_entity_ids": [",".join(predictions[x]) for x in predictions]}).to_csv(outdir / "matching_results.tsv", sep="\t", index=False)
    pd.DataFrame({"source1_entity_id": list(test_cmap),
                  "candidate_entity_ids": [",".join(sorted(set(test_cmap[x]))) for x in test_cmap]}).to_csv(outdir / "candidate_pairs.tsv", sep="\t", index=False)
    report = {"validation_f05": best_score, "threshold": best_threshold,
              "train_source1": len(s1), "train_candidates": len(keys_all),
              "test_source1": len(t1), "test_candidates": len(test_keys),
              "positive_training_pairs": int(y_all.sum())}
    (outdir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
