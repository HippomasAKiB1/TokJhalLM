"""Bengali-script text helpers and the T3 rule-based romanizer (Prompt 3, D7-amended).

T3 is the deterministic baseline and safety net of the transliteration engine. It maps
graphemes to the colloquial Banglish convention used by Vashantor (ভ -> v, ফ -> f,
ছ/চ -> ch, য় -> y) and handles the inherent vowel with a schwa-deletion heuristic:

- a consonant with no vowel sign carries the inherent "o", except
  - at the end of a word, and
  - in a V C . C V pattern: the consonant is preceded by a vowel AND the next
    consonant carries an explicit vowel sign (করবো -> korbo, কথা -> kotha).
- hasanta (্) suppresses the inherent vowel; ্য and ্ব are absorbed after a
  consonant; ্র becomes "r".

Also here: segmenting text into Bengali words / other runs, digit and danda
normalization, and the word normalization shared by T1/T2/T3 and evaluation.

Run standalone for a demo:
    E:\\Anaconda\\envs\\tokjhallm\\python.exe -m data.translit_rules
"""

from __future__ import annotations

import re
import unicodedata

BENGALI_WORD_RE = re.compile(r"[\u0980-\u09FF\u200c\u200d]+")
SEGMENT_RE = re.compile(r"[\u0980-\u09FF\u200c\u200d]+|[^\u0980-\u09FF\u200c\u200d]+")
ZW_RE = re.compile("[\u200c\u200d]")
BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
EVAL_WORD_RE = re.compile(r"[a-z0-9]+")

HASANTA = "\u09cd"
NUKTA = "\u09bc"

CONSONANTS: dict[str, str] = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "ch", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "f", "ব": "b", "ভ": "v", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh",
    "স": "s", "হ": "h", "ৎ": "t",
}
# consonant + nukta (NFC keeps these decomposed)
NUKTA_CONSONANTS: dict[str, str] = {"ড": "r", "ঢ": "rh", "য": "y"}
VOWELS: dict[str, str] = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
}
VOWEL_SIGNS: dict[str, str] = {
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
}
MODIFIERS: dict[str, str] = {"ং": "ng", "ঃ": "h", "ঁ": ""}
PUNCT = str.maketrans({"।": ".", "॥": "."})


def normalize_bengali_word(word: str) -> str:
    """Lookup key for a Bengali word: NFC, zero-width joiners removed."""
    return ZW_RE.sub("", unicodedata.normalize("NFC", word))


def normalize_roman(text: str) -> str:
    """Romanized text for lexicon values and evaluation: lowercase, ASCII digits."""
    return text.lower().translate(BENGALI_DIGITS)


def eval_words(text: str) -> list[str]:
    """Words for WER/chrF: lowercase, ASCII digits, punctuation stripped."""
    return EVAL_WORD_RE.findall(normalize_roman(text))


def source_words(text: str) -> list[str]:
    """Bengali-side words for alignment with eval_words (digits as ASCII words)."""
    t = unicodedata.normalize("NFC", text).translate(BENGALI_DIGITS)
    return [normalize_bengali_word(w) for w in re.findall(r"[\u0980-\u09FF\u200c\u200d]+|[A-Za-z0-9]+", t)]


def segments(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_bengali_word, run) segments covering the whole string."""
    t = unicodedata.normalize("NFC", text)
    return [(bool(BENGALI_WORD_RE.fullmatch(m)), m) for m in SEGMENT_RE.findall(t)]


def finalize_other(run: str) -> str:
    """Non-Bengali runs: Bengali digits/danda to ASCII, everything else unchanged."""
    return run.translate(BENGALI_DIGITS).translate(PUNCT)


def _units(word: str) -> list[tuple[str, str]]:
    """Parse a word into (kind, roman) units: C consonant, V vowel/sign, H hasanta, M modifier."""
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(word):
        ch = word[i]
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in CONSONANTS:
            if nxt == NUKTA and ch in NUKTA_CONSONANTS:
                out.append(("C", NUKTA_CONSONANTS[ch]))
                i += 2
                continue
            out.append(("C", CONSONANTS[ch]))
        elif ch in VOWELS:
            out.append(("V", VOWELS[ch]))
        elif ch in VOWEL_SIGNS:
            out.append(("S", VOWEL_SIGNS[ch]))
        elif ch == HASANTA:
            out.append(("H", ""))
        elif ch in MODIFIERS:
            out.append(("M", MODIFIERS[ch]))
        elif "০" <= ch <= "৯":
            out.append(("D", ch.translate(BENGALI_DIGITS)))
        # other marks (nukta alone, rare signs) are dropped
        i += 1
    return out


def romanize_rules(word: str) -> str:
    """T3: deterministic romanization of one Bengali word."""
    units = _units(normalize_bengali_word(word))
    out: list[str] = []
    n = len(units)
    prev_vowel = False  # did the previous emitted syllable end in a vowel?
    i = 0
    while i < n:
        kind, rom = units[i]
        if kind != "C":
            out.append(rom)
            prev_vowel = kind in ("V", "S")
            i += 1
            continue
        nk = units[i + 1][0] if i + 1 < n else ""
        if nk == "H":
            after = units[i + 2] if i + 2 < n else ("", "")
            # phala forms: ্য (j) and ্ব (b) are absorbed; ্র stays as r
            if after[0] == "C" and after[1] in ("j", "b") and i > 0:
                out.append(rom)
                i += 3
                prev_vowel = False
                if i < n and units[i][0] in ("S", "V"):
                    continue
                if i >= n:
                    break
                out.append("o")
                prev_vowel = True
                continue
            out.append(rom)
            prev_vowel = False
            i += 2
            continue
        if nk == "S":
            out.append(rom)
            i += 1
            continue
        # bare consonant: decide inherent vowel
        out.append(rom)
        last = i == n - 1 or all(k == "M" for k, _ in units[i + 1:])
        next_has_sign = (nk == "C" and i + 2 < n and units[i + 2][0] == "S")
        if last and i > 0:
            prev_vowel = False
        elif prev_vowel and next_has_sign:
            prev_vowel = False
        else:
            out.append("o")
            prev_vowel = True
        i += 1
    return "".join(out)


def romanize_text_rules(text: str) -> str:
    """T3 over running text (Bengali words romanized, everything else kept)."""
    return "".join(romanize_rules(s) if is_b else finalize_other(s) for is_b, s in segments(text))


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    for w in ["আমার", "ভালো", "কেমন", "করবো", "কথা", "তুমি", "বাংলাদেশ", "বিদ্যালয়", "পড়াশোনা", "২০২৪", "স্বাধীনতা"]:
        print(f"{w} -> {romanize_rules(w)}")
    print(romanize_text_rules("আমি ২০২৪ সালে ঢাকায় ছিলাম। OK?"))
