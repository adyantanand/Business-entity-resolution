from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from text import digit_tokens, tokens


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(a: pd.Series, b: pd.Series) -> dict[str, float]:
    an, bn = a["_name"], b["_name"]
    aa, ba = a["_address"], b["_address"]
    name_tokens_a, name_tokens_b = a["_name_tokens"], b["_name_tokens"]
    addr_tokens_a, addr_tokens_b = a["_address_tokens"], b["_address_tokens"]
    digits = a["_digits"] & b["_digits"]
    return {
        "name_ratio": fuzz.ratio(an, bn) / 100.0 if an and bn else 0.0,
        "name_token_set": fuzz.token_set_ratio(an, bn) / 100.0 if an and bn else 0.0,
        "name_jaccard": jaccard(name_tokens_a, name_tokens_b),
        "name_contains": float(bool(an and bn and (an in bn or bn in an))),
        "address_ratio": fuzz.ratio(aa, ba) / 100.0 if aa and ba else 0.0,
        "address_token_set": fuzz.token_set_ratio(aa, ba) / 100.0 if aa and ba else 0.0,
        "address_jaccard": jaccard(addr_tokens_a, addr_tokens_b),
        "shared_digits": float(len(digits)),
        "country_equal": float(bool(a["_country"] and a["_country"] == b["_country"])),
        "country_compatible": float(not a["_country"] or not b["_country"] or a["_country"] == b["_country"]),
        "name_missing": float(not an or not bn),
        "address_missing": float(not aa or not ba),
        "name_length_delta": float(abs(len(an) - len(bn))),
        "address_length_delta": float(abs(len(aa) - len(ba))),
    }


def make_pairs(left: pd.DataFrame, right: pd.DataFrame, cmap: dict[str, list[str]]) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    lookup = right.set_index("entity_id", drop=False)
    rows, keys = [], []
    for _, a in left.iterrows():
        for rid in cmap.get(a["entity_id"], []):
            if rid not in lookup.index:
                continue
            b = lookup.loc[rid]
            rows.append(pair_features(a, b))
            keys.append((a["entity_id"], rid))
    return pd.DataFrame(rows).fillna(0.0), keys
