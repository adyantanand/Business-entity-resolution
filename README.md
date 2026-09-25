# Business Entity Resolution

End-to-end, local-only pipeline for the Business Entity Resolution Challenge. It uses only the supplied TSV files: normalization, multi-rule candidate blocking, pairwise similarity features, a supervised classifier, validation threshold selection for macro F0.5, and submission generation.

## Expected data layout

```text
dataset/train/train_source1.tsv
dataset/train/train_source2.tsv
dataset/train/train_source3.tsv
dataset/train/train_ground_truth.tsv
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

## Run

From the repository root:

```bash
python -m pip install -r requirements.txt
python code/business_entity_resolution/src/run_pipeline.py --data-root . --output-dir output
```

The command writes `output/matching_results.tsv`, `output/candidate_pairs.tsv`, and `output/validation_report.json`.

If the supplied challenge validator exists, run it separately:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

No external databases, APIs, geocoders, or internet lookups are used.
