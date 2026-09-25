# Methodology

## Method

This submission uses a local-only, supervised entity-resolution pipeline. It does not use external databases, APIs, geocoding, or internet lookups.

Names and addresses are Unicode-normalized, lowercased, punctuation-normalized, whitespace-normalized, and processed with a small set of generic legal/address abbreviation rules. The original country label is treated as an open-set string and is never restricted to a fixed list.

## Candidate generation

Candidates are the union of inverted-index blocks on informative name tokens, address tokens, shared digit tokens, and approximate top-k name/address matches. Country compatibility is used to prefer compatible candidates without dropping records whose country is missing. Training positives from the supplied ground truth are forced into the training candidate set so the classifier sees every labeled positive. The final candidate set written to `candidate_pairs.tsv` is exactly the set scored during test inference.

## Features and model

Each candidate pair receives name edit/token-set/Jaccard/containment features, address edit/token-set/Jaccard features, shared numeric-token count, country compatibility/equality, missingness indicators, and length differences. A balanced random forest is trained on positive and blocked-negative pairs.

Validation is split by Source 1 entity rather than by pair, preventing leakage between pairs belonging to the same entity. A threshold grid is selected using macro F0.5, with higher thresholds preferred on ties because the challenge penalizes false merges more heavily. The model is then refit on all training candidates and applied to test candidates.

## Reproduction

From the repository root, install `requirements.txt` and run:

```bash
python code/business_entity_resolution/src/run_pipeline.py --data-root . --output-dir output
```

The two required TSV files are created in `output/`. The pipeline preserves every test Source 1 entity, permits multiple matches, and ensures final matches are selected from the candidate set.
