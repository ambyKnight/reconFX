"""
Tests for ingestion, reference token extraction, and candidate blocking.
"""

from collections import namedtuple
import pandas as pd
import pytest

from recon.blocking import collision_counts, get_candidate_pool, index_ledger
from recon.decide import choose
from recon.ingest import get_dataset_dir, tokenize_references


def test_tokenize_references():
    assert tokenize_references("INV-12345/ABCD") == {"12345", "ABCD"}
    assert tokenize_references("NO TOK") == set()
    assert tokenize_references("SHORT") == {"SHORT"}
    assert tokenize_references(None) == set()
    assert tokenize_references("") == set()


def test_index_ledger():
    ledger = pd.DataFrame([
        {"A_id": "L1", "amt": 100.50, "A_allocation": "ALLOC_1"},
        {"A_id": "L2", "amt": 100.50, "A_allocation": "ALLOC_2"},
        {"A_id": "L3", "amt": 250.00, "A_allocation": "ALLOC_3"},
    ])
    by_amt = index_ledger(ledger)
    assert len(by_amt[100.50]) == 2
    assert len(by_amt[250.00]) == 1
    assert len(by_amt[999.00]) == 0


def test_choose_logic():
    Row = namedtuple("Row", ["A_allocation", "tokens"])
    b_tokens = {"INV9999", "CUST"}
    BRow = namedtuple("BRow", ["tokens"])
    b = BRow(tokens=b_tokens)

    # 1. Single allocation -> confident 0.95
    c1 = [Row(A_allocation="ALLOC_A", tokens=set()), Row(A_allocation="ALLOC_A", tokens=set())]
    alloc, conf, tier = choose(c1, b)
    assert alloc == "ALLOC_A"
    assert conf == 0.95
    assert tier == "single allocation"

    # 2. Multiple allocations broken by reference tokens -> medium 0.80
    c2 = [
        Row(A_allocation="ALLOC_A", tokens={"INV9999"}),
        Row(A_allocation="ALLOC_B", tokens={"OTHER"}),
    ]
    alloc, conf, tier = choose(c2, b)
    assert alloc == "ALLOC_A"
    assert conf == 0.80
    assert tier == "reference tokens break the tie"

    # 3. Ambiguous allocations -> low 0.35
    c3 = [
        Row(A_allocation="ALLOC_A", tokens=set()),
        Row(A_allocation="ALLOC_B", tokens=set()),
    ]
    alloc, conf, tier = choose(c3, b)
    assert conf == 0.35
    assert "ambiguous" in tier

    # 4. No candidates -> 0.0
    alloc, conf, tier = choose([], b)
    assert alloc is None
    assert conf == 0.0
    assert tier == "no candidate"


def test_dataset_dir_cache():
    # Verify cached dataset directory is resolved without network
    d = get_dataset_dir()
    assert d.exists()
    assert (d / "BenchRec_cash_v1.0_eval.csv").exists()
