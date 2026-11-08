"""Tests for Prompt 3 Stage A (D16 NSFW fix) and Stage 0 (transliteration + metrics)."""

from __future__ import annotations

from collections import Counter

import pytest

from data.clean import nsfw_reason, nsfw_url_hits, url_tokens
from data.common import load_config, repo_path
from data.translit import Transliterator
from data.translit_eval import band, corpus_chrf, corpus_wer
from data.translit_rules import eval_words, romanize_rules, romanize_text_rules, source_words

HAS_VASHANTOR = repo_path("data/raw/vashantor/Vashantor_CSV_Format.zip").is_file()


# --- D16 NSFW fix ----------------------------------------------------------

def test_url_rule_uses_tokens_not_substrings() -> None:
    # the Prompt 2 ben_Beng false positives fired on 'sex'/'chud' inside other words
    for url in ["https://jagobangla.in/chandrachud-x/","http://newsexpressdigital.in/bengali-breaking-news/",
                "https://jagobangla.in/chandrachud-criticizes-the-bulldozer-policy/",
                "https://www.thewall.in/news/the-world-is-having-far-less-sex-using-condoms/",
                "https://www.adulteducation.org/x"]:
        assert nsfw_url_hits(url) == [], url
    # glued domain words still fire at a token edge
    assert nsfw_url_hits("https://cosplayfreeporn.com/bn/video/1") == ["porn"]
    assert nsfw_url_hits("http://www.banglachoti.in/page/60/") == ["choti"]
    assert nsfw_url_hits("http://x.blogspot.com/maa-ke-chodar-protishod.html") == ["chod"]
    assert nsfw_url_hits("http://example.com/adult/chat") == ["adult"]
    # replay regressions: porn mid-token, chud as a slug prefix (never as a suffix)
    assert nsfw_url_hits("https://prostopornohd.net/bn/video/1") == ["porn"]
    assert nsfw_url_hits("http://x.blogspot.com/2010/06/ochena-boudike-chudlam.html") == ["chud"]


def test_url_tokens_ignore_percent_encoding() -> None:
    assert url_tokens("https://a.com/%E0%A6%AC-porn") == ["a", "com", "porn"]


def test_bengali_news_kept_explicit_dropped() -> None:
    news = "আজ ধর্ষণের ঘটনায় যৌন নির্যাতনের অভিযোগে থানায় মামলা হয়েছে বলে পুলিশ জানিয়েছে"
    assert nsfw_reason(news) is None
    assert nsfw_reason("চোদা গুদ " + "শব্দ " * 50) == "bengali_terms"
    assert nsfw_reason("চোদা " + "শব্দ " * 50) is None  # one distinct term is not enough
    assert nsfw_reason("সে মাগী খানকি বলে গালি দিয়েছে বলে অভিযোগ " * 3) is None  # quoted slurs in news


def test_nsfw_config_has_no_removed_url_terms() -> None:
    n = load_config()["nsfw"]
    assert "sex" not in n["url_tokens_exact"] + n["url_tokens_affix"]
    assert "chud" not in n["url_tokens_exact"] + n["url_tokens_affix"]


# --- metrics ---------------------------------------------------------------

def test_chrf_identity_disjoint_and_hand_computed() -> None:
    assert corpus_chrf(["amar valo"], ["amar valo"]) == pytest.approx(1.0)
    assert corpus_chrf(["xyz"], ["abc"]) == 0.0
    # n=1: P=1, R=2/3; n=2: P=1, R=1/2 -> P=1, R=7/12; beta=1 -> F = 14/19
    assert corpus_chrf(["ab"], ["abc"], max_order=2, beta=1.0) == pytest.approx(14 / 19)


def test_chrf_ignores_whitespace() -> None:
    assert corpus_chrf(["ami valo"], ["amivalo"]) == pytest.approx(1.0)


def test_wer() -> None:
    assert corpus_wer([["a", "b"]], [["a", "c", "b"]]) == pytest.approx(1 / 3)
    assert corpus_wer([["a"], ["x", "y"]], [["a"], ["x", "z"]]) == pytest.approx(1 / 3)


def test_gate_bands_match_preregistered_thresholds() -> None:
    gate = load_config(repo_path("configs/translit.yaml"))["gate"]
    assert band(0.19, 0.71, gate) == "pass"
    assert band(0.20, 0.71, gate) == "middle"
    assert band(0.19, 0.70, gate) == "middle"
    assert band(0.30, 0.90, gate) == "stop"
    assert band(0.10, 0.54, gate) == "stop"


# --- rules + routing -------------------------------------------------------

@pytest.mark.parametrize("bn,roman", [("আমার", "amar"), ("ভালো", "valo"), ("কেমন", "kemon"),
                                      ("করবো", "korbo"), ("কথা", "kotha"), ("তুমি", "tumi"), ("বাংলাদেশ", "bangladesh")])
def test_rules_common_words(bn: str, roman: str) -> None:
    assert romanize_rules(bn) == roman


def test_rules_keep_non_bengali_and_convert_digits() -> None:
    assert romanize_text_rules("আমি ২০২৪ সালে OK 😍।") == "ami 2024 sale OK 😍."


def test_eval_and_source_words_align() -> None:
    assert eval_words("Tomar abbu kemon ache?") == ["tomar", "abbu", "kemon", "ache"]
    assert source_words("তোমার আব্বু কেমন আছে?") == ["তোমার", "আব্বু", "কেমন", "আছে"]


def test_transliterator_routes_t1_then_fallback() -> None:
    lex = {"ভালো": Counter({"valo": 9, "bhalo": 1})}
    calls: list[list[str]] = []

    def fb(ws: list[str]) -> list[str]:
        calls.append(ws)
        return ["X" for _ in ws]

    tr = Transliterator(lex, fb, seed=1)
    assert tr.romanize_text("ভালো বই, ভালো!") == "valo X, valo!"
    assert calls == [["বই"]]
    assert tr.tier("ভালো") == "T1" and tr.tier("বই") == "fallback"
    sampled = {tr.romanize_words(["ভালো"], sample=True)[0] for _ in range(200)}
    assert sampled == {"valo", "bhalo"}


@pytest.mark.skipif(not HAS_VASHANTOR, reason="raw Vashantor data not present (CI)")
def test_lexicon_never_reads_test_split() -> None:
    from data.translit import build_lexicon, load_translit_config

    cfg = load_translit_config()
    assert "Test" not in cfg["lexicon"]["splits"]
    bad = {**cfg, "lexicon": {**cfg["lexicon"], "splits": ["Train", "Test"]}}
    with pytest.raises(ValueError):
        build_lexicon(bad)


# --- T2 model ---------------------------------------------------------------

def test_char_vocab_and_model_shapes() -> None:
    import torch

    from data.translit_model import CharSeq2Seq, CharVocab, _pad, greedy_decode

    v = CharVocab.from_words(["আমি", "ami"])
    assert v.decode(v.encode("ami", 10)) == "ami"
    assert v.decode(v.encode("ami", 10, bos=True)) == "ami"
    mcfg = {"d_model": 32, "n_heads": 2, "enc_layers": 1, "dec_layers": 1, "ffn": 64, "dropout": 0.0,
            "max_src_chars": 8, "max_tgt_chars": 8}
    torch.manual_seed(0)
    m = CharSeq2Seq(len(v), len(v), mcfg)
    logits = m(_pad([v.encode("আমি", 8)]), _pad([v.encode("ami", 9, bos=True)])[:, :-1])
    assert logits.shape == (1, 4, len(v))
    out = greedy_decode(m, ["আমি", "মি"], v, v, mcfg, "cpu")
    assert len(out) == 2 and all(isinstance(s, str) for s in out)
