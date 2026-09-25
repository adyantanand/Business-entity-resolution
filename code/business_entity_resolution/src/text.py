from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd

LEGAL = {
    "corporation": "corp", "incorporated": "inc", "company": "co",
    "limited": "ltd", "private limited": "pvt ltd", "privatelimited": "pvt ltd",
}
ADDRESS = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "lane": "ln", "drive": "dr", "highway": "hwy", "apartment": "apt",
    "building": "bldg", "floor": "fl",
}
STOP = {"the", "and", "or", "of", "co", "corp", "inc", "ltd", "limited", "pvt", "private", "llp", "company"}


def normalize(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("&", " and ")
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value: object) -> str:
    text = normalize(value)
    # Long phrases first so that private limited is handled consistently.
    for old, new in sorted(LEGAL.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_address(value: object) -> str:
    text = normalize(value)
    for old, new in ADDRESS.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str, informative: bool = False) -> set[str]:
    result = set(text.split())
    return {x for x in result if len(x) > 1 and (not informative or x not in STOP)}


def digit_tokens(text: str) -> set[str]:
    return {x for x in text.split() if any(c.isdigit() for c in x) and len(x) >= 2}
