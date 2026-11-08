"""Offline tests for the Prompt 2 data pipeline (no network, synthetic text only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.audit import gate_exit_code, gating_hits, macro_f1_constant
from data.clean import clean_text, normalize_for_match, nsfw_reason, script_stats, word_script_counts
from data.common import load_config
from data.leakage import LeakageIndex
from data.minhash import MinHasher, jaccard, near_dup_clusters, shingles
from data.readers import iter_json_array
from data.splits import Row, assign_splits, dedup_exact


@pytest.fixture(scope="module")
def cfg() -> dict:
    """The real configs/data.yaml."""
    return load_config()


# --- clean ---------------------------------------------------------------

def test_clean_strips_urls_mentions_keeps_emoji_and_zwnj() -> None:
    text = "Darun 😍 https://x.co/abc @rahim​ ভাল‌ো"
    out = clean_text(text)
    assert "https" not in out and "@rahim" not in out and "​" not in out
    assert "😍" in out
    assert "‌" in out  # ZWNJ is meaningful in Bengali and must survive


def test_normalize_for_match_is_aggressive() -> None:
    assert normalize_for_match("Khub VALO laglo!!! 😍😍") == "khub valo laglo"
    assert normalize_for_match("sooooo good") == "soo good"
    assert normalize_for_match("আমি  ভালো আছি।") == "আমি ভালো আছি"


def test_script_detection() -> None:
    assert script_stats("amar mon valo nei").label == "latin"
    assert script_stats("আমার মন ভালো নেই").label == "bengali"
    assert script_stats("amar অ্যালার্জি nei, kintu khub kosto").label in ("latin", "mixed")
    assert script_stats("123 !!!").label == "none"
    assert word_script_counts("valo ভালো 42") == {"bengali": 1, "latin": 1, "mixed": 0, "other": 1}


def test_nsfw_filter_uses_url_and_unambiguous_terms() -> None:
    assert nsfw_reason("normal text here", "http://example-choti-site.com/x") == "url"
    assert nsfw_reason("ei mal ta gud, dam o kom") is None  # ambiguous words are not terms
    assert nsfw_reason("ami take choda er golpo bolbo") == "terms"


# --- readers -------------------------------------------------------------

def test_iter_json_array_streams_across_chunk_boundaries(tmp_path: Path) -> None:
    items = [{"Reviews": [{"Review Content": f"review {i} " + "x" * (i * 7)}]} for i in range(200)]
    path = tmp_path / "big.json"
    path.write_text(json.dumps(items, indent=4, ensure_ascii=False), encoding="utf-8")
    got = list(iter_json_array(path, chunk_chars=97))  # tiny chunks force refills mid-object
    assert got == items


def test_iter_json_array_rejects_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('[{"a": 1}, {"b": ', encoding="utf-8")
    with pytest.raises(ValueError):
        list(iter_json_array(path, chunk_chars=8))


# --- minhash + splits ----------------------------------------------------

def test_near_dup_clusters_join_close_texts_only() -> None:
    hasher = MinHasher(num_perm=128, bands=32, seed=1)
    texts = [
        "product ta khub valo laglo dam o kom delivery fast",
        "product ta khub valo laglo dam o kom delivery fast!",
        "network khub baje atar moto faltu net r nai",
    ]
    norm = [normalize_for_match(t) for t in texts]
    clusters = near_dup_clusters(norm, 5, hasher, 0.8)
    assert clusters[0] == clusters[1] != clusters[2]
    assert jaccard(shingles(norm[0], 5), shingles(norm[1], 5)) == 1.0


def test_dedup_drops_conflicting_labels_and_collapses_duplicates() -> None:
    rows = [(0, "Khub valo", 0), (1, "khub valo!!", 0), (2, "Faltu", 1), (3, "faltu", 2), (4, "Thik ache", 2)]
    kept, stats = dedup_exact(rows)
    assert [r.row_id for r in kept] == [0, 4]
    assert stats["conflicting_label_groups"] == 1 and stats["rows_dropped_conflicting"] == 2
    assert stats["rows_collapsed"] == 1


def test_assign_splits_keeps_clusters_together_and_stratifies() -> None:
    # pairs of near-duplicates; each pair shares a label (conflicting groups are dropped upstream)
    rows = [Row(i, f"t{i}", (i // 2) % 4, f"t{i}", str(i)) for i in range(400)]
    clusters = [i // 2 for i in range(400)]
    out = assign_splits(rows, clusters, n_folds=10, seed=42)
    assert {k: len(v) for k, v in out.items()} == {"train": 320, "dev": 40, "test": 40}
    where = {r.row_id: name for name, rs in out.items() for r in rs}
    assert all(where[2 * c] == where[2 * c + 1] for c in range(200))
    for name in ("dev", "test"):
        labels = [r.label for r in out[name]]
        assert all(labels.count(lab) == 10 for lab in range(4))


# --- leakage -------------------------------------------------------------

@pytest.fixture()
def index(cfg: dict) -> LeakageIndex:
    ix = LeakageIndex(cfg)
    ix.add("test", 1, 0, "Vaia apnar voice ta onek shundor, ami protidin shuni")
    ix.add("dev", 2, 1, "Network khub baje")
    ix.add("train", 3, 2, "Graphics card change kore intel arc nile total koto porbe")
    return ix


def test_leakage_detects_exact_containment_and_near_dup(index: LeakageIndex) -> None:
    def kinds(doc: str) -> set[tuple[str, str]]:
        return {(h.split, h.match) for h in index.scan(doc, "demo", "d")}

    assert kinds("Ei product ta khub valo, delivery o fast chilo") == set()
    assert ("test", "exact") in kinds("VAIA apnar voice ta onek shundor ami protidin shuni!!!")
    long_doc = "Intro words go here and " * 5 + "vaia apnar voice ta onek shundor ami protidin shuni " + "and more " * 30
    assert ("test", "containment") in kinds(long_doc)
    assert ("dev", "exact") in kinds("Some intro line\nNetwork khub baje\n" + "filler text " * 20)
    assert ("test", "near_dup") in kinds("vaia apnar voice ta onek sundor ami protidin shuni")


def test_gate_fails_on_planted_dev_test_leak_only(index: LeakageIndex) -> None:
    cfg = index.cfg
    checked = set(cfg["leakage"]["splits_checked"])
    train_only = [h.as_dict() for h in index.scan("graphics card change kore intel arc nile total koto porbe", "demo", "a")]
    assert train_only and gate_exit_code(train_only, checked, cfg) == 0  # train overlap is not leakage
    planted = [h.as_dict() for h in index.scan("network khub baje", "demo", "b")]
    assert gate_exit_code(planted, checked, cfg) == 1
    soft = [{"split": "test", "row_id": 9, "match": "near_dup_soft", "sample_text": "x y z", "source": "s", "doc_id": "d"}]
    assert gating_hits(soft, checked, cfg) == []


def test_exclusion_list_skips_doc_and_rejects_changed_text() -> None:
    import hashlib

    from data.corpus import is_excluded

    text = "atto valo laglo na"
    excl = {("banglishrev", "71425:32"): hashlib.sha256(text.encode("utf-8")).hexdigest()}
    assert is_excluded("banglishrev", "71425:32", text, excl)
    assert not is_excluded("banglishrev", "1:1", text, excl)
    with pytest.raises(RuntimeError):
        is_excluded("banglishrev", "71425:32", text + " edited", excl)


def test_exclusion_covers_normalized_copies(cfg: dict) -> None:
    from data.corpus import load_excluded_norm_keys, norm_key

    keys = load_excluded_norm_keys(cfg)
    # approved doc 'Masha Allah!\nkhub valo' and a later raw copy normalize identically
    assert norm_key(normalize_for_match("masha allah❤️❤️❤️khub valo....")) in keys
    assert norm_key(normalize_for_match("ei product ta khub valo")) not in keys


def test_committed_exclusion_list_is_well_formed(cfg: dict) -> None:
    from data.common import repo_path

    data = json.loads(repo_path(cfg["paths"]["leakage_exclusions"]).read_text(encoding="utf-8"))
    assert data["approved_by"] == "Akib"
    for d in data["docs"]:
        assert {"source", "doc_id", "sha256", "reason", "text"} <= set(d)
        assert len(d["sha256"]) == 64


# --- D14 specificity gate ------------------------------------------------

def _hits(text: str, n_docs: int, split: str = "test", row_id: int = 1) -> list[dict]:
    """`n_docs` exact hits of one sample in distinct pretraining documents."""
    return [
        {"split": split, "row_id": row_id, "sample_text": text, "source": "fineweb2_ben_latn",
         "doc_id": f"doc{i}", "match": "exact", "score": 1.0, "excerpt": text}
        for i in range(n_docs)
    ]


def test_d14_allowlisted_greeting_in_10_docs_does_not_gate(cfg: dict) -> None:
    hits = _hits("Assalamu alaikum", 10)
    assert gate_exit_code(hits, {"dev", "test"}, cfg) == 0


def test_d14_distinctive_6_word_phrase_in_1_doc_gates(cfg: dict) -> None:
    hits = _hits("Restaurant ta fancy, but food oily", 1)
    assert gate_exit_code(hits, {"dev", "test"}, cfg) == 1


def test_d14_8_word_phrase_in_6_docs_does_not_gate(cfg: dict) -> None:
    assert int(cfg["leakage"]["formulaic"]["min_doc_frequency"]) <= 6
    hits = _hits("ei product ta amar khub kaje lagse vai", 6)
    assert gate_exit_code(hits, {"dev", "test"}, cfg) == 0


def test_d14_8_word_phrase_in_2_docs_gates(cfg: dict) -> None:
    hits = _hits("ei product ta amar khub kaje lagse vai", 2)
    assert gate_exit_code(hits, {"dev", "test"}, cfg) == 1


def test_majority_baseline_macro_f1() -> None:
    labels = [2] * 33 + [0] * 27 + [1] * 31 + [3] * 9
    f1 = macro_f1_constant(labels, predict=2, classes=[0, 1, 2, 3])
    assert f1 == pytest.approx((2 * 0.33 / 1.33) / 4, abs=1e-3)
