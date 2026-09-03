"""
Shared syntactic feature extraction — mirrors the logic in
Human_vs_AI_Syntax_Corpus.ipynb so the Streamlit app and the notebook
produce identical numbers.
"""
import re

import numpy as np
import pandas as pd
import spacy
import streamlit as st

CLAUSE_DEPS = {"advcl", "ccomp", "xcomp", "relcl", "acl"}

FEATURES = [
    "sentence_length", "tree_depth", "n_clauses", "avg_dep_distance",
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pron_ratio", "passive",
]

FEATURE_LABELS = {
    "sentence_length": "Sentence length (tokens)",
    "tree_depth": "Dependency tree depth",
    "n_clauses": "Clause count",
    "avg_dep_distance": "Avg. dependency distance",
    "noun_ratio": "Noun ratio",
    "verb_ratio": "Verb ratio",
    "adj_ratio": "Adjective ratio",
    "adv_ratio": "Adverb ratio",
    "pron_ratio": "Pronoun ratio",
    "passive": "Passive voice rate",
}


@st.cache_resource(show_spinner="Loading spaCy model (first run only)...")
def load_nlp():
    """Load en_core_web_sm, downloading it on first run if missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tree_depth(spacy_sent) -> int:
    """Max distance from any token up to the sentence root."""
    max_depth = 0
    for token in spacy_sent:
        depth = 0
        node = token
        while node.head != node:
            depth += 1
            node = node.head
        max_depth = max(max_depth, depth)
    return max_depth


def extract_features(spacy_sent):
    """Compute the full syntactic feature set for one spaCy sentence span."""
    tokens = [t for t in spacy_sent if not t.is_punct and not t.is_space]
    n_tokens = len(tokens)
    if n_tokens == 0:
        return None

    pos_counts = {}
    for t in tokens:
        pos_counts[t.pos_] = pos_counts.get(t.pos_, 0) + 1

    n_clauses = 1 + sum(1 for t in spacy_sent if t.dep_ in CLAUSE_DEPS)
    is_passive = any(t.dep_ in ("nsubjpass", "auxpass") for t in spacy_sent)
    dep_distances = [abs(t.i - t.head.i) for t in spacy_sent if t.head != t]
    avg_dep_distance = float(np.mean(dep_distances)) if dep_distances else 0.0

    return {
        "sentence_length": n_tokens,
        "tree_depth": tree_depth(spacy_sent),
        "n_clauses": n_clauses,
        "avg_dep_distance": avg_dep_distance,
        "passive": int(is_passive),
        "noun_ratio": pos_counts.get("NOUN", 0) / n_tokens,
        "verb_ratio": pos_counts.get("VERB", 0) / n_tokens,
        "adj_ratio": pos_counts.get("ADJ", 0) / n_tokens,
        "adv_ratio": pos_counts.get("ADV", 0) / n_tokens,
        "pron_ratio": pos_counts.get("PRON", 0) / n_tokens,
    }


def analyze_text(nlp, text: str, label: str = "input") -> pd.DataFrame:
    """Sentence-split + annotate a raw text block. Returns a feature DataFrame."""
    doc = nlp(text)
    rows = []
    for sent in doc.sents:
        clean = sent.text.strip()
        if len(clean.split()) < 3:
            continue
        feats = extract_features(sent)
        if feats is None:
            continue
        feats["sentence"] = clean
        feats["source"] = label
        rows.append(feats)
    return pd.DataFrame(rows)
