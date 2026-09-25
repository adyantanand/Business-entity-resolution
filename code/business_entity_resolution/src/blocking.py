from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd
from rapidfuzz import fuzz, process

from text import digit_tokens, tokens


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().fillna("")
    out["_name"] = out["business_name"].astype(str)
    out["_address"] = out["business_address"].astype(str)
    out["_country"] = out["country"].astype(str).str.strip().str.casefold()
    out["_name_tokens"] = out["_name"].map(lambda x: tokens(x, True))
    out["_address_tokens"] = out["_address"].map(lambda x: tokens(x, False))
    out["_digits"] = (out["_name"] + " " + out["_address"]).map(digit_tokens)
    return out


def candidate_map(left: pd.DataFrame, right: pd.DataFrame, top_k: int = 8, max_candidates: int = 120,
                  forced: dict[str, set[str]] | None = None) -> dict[str, list[str]]:
    """Multi-pass blocking. The returned IDs are exactly the candidates scored later."""
    right = right.reset_index(drop=True)
    ids = right["entity_id"].tolist()
    name_choices = {i: right.loc[i, "_name"] for i in range(len(right))}
    addr_choices = {i: right.loc[i, "_address"] for i in range(len(right))}
    by_name: dict[str, set[int]] = defaultdict(set)
    by_addr: dict[str, set[int]] = defaultdict(set)
    by_digit: dict[str, set[int]] = defaultdict(set)
    for i, row in right.iterrows():
        for t in row["_name_tokens"]:
            by_name[t].add(i)
        for t in row["_address_tokens"]:
            by_addr[t].add(i)
        for t in row["_digits"]:
            by_digit[t].add(i)

    result: dict[str, list[str]] = {}
    for _, row in left.iterrows():
        found: set[int] = set()
        for t in row["_name_tokens"]:
            found.update(by_name.get(t, set()))
        for t in row["_address_tokens"]:
            found.update(by_addr.get(t, set()))
        for t in row["_digits"]:
            found.update(by_digit.get(t, set()))
        country = row["_country"]
        # Approximate fallbacks are essential for typo/transliteration cases.
        if row["_name"]:
            found.update(x[2] for x in process.extract(row["_name"], name_choices, scorer=fuzz.WRatio, limit=top_k, score_cutoff=45))
        if row["_address"]:
            found.update(x[2] for x in process.extract(row["_address"], addr_choices, scorer=fuzz.WRatio, limit=top_k, score_cutoff=40))
        # Prefer country-compatible candidates, but do not discard open-set or missing-country records.
        compatible = {i for i in found if not country or not right.loc[i, "_country"] or right.loc[i, "_country"] == country}
        if compatible:
            found = compatible
        if forced and row["entity_id"] in forced:
            wanted = forced[row["entity_id"]]
            found.update(i for i, entity_id in enumerate(ids) if entity_id in wanted)
        # Deterministic ordering makes files reproducible and bounds pathological blocks.
        ordered = sorted(found, key=lambda i: ids[i])[:max_candidates]
        result[row["entity_id"]] = [ids[i] for i in ordered]
    return result
