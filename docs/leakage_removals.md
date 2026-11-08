# Leakage removals

Pretraining documents removed because they matched BnSentMix dev/test samples under the D14 specificity gate.
Approved by Akib on 2026-11-08. The test set and split manifests were **not** modified.

Enforcement: the machine-readable list is `data/leakage_exclusions.json`. `data/corpus.py` drops each listed
(source, doc_id) on every rebuild and raises if a listed document's sha256 no longer matches, so the removal
is reproducible and cannot silently drift.

Total: 6 documents, 26 words, all from BanglishRev (out of ~185M corpus words): the 5 approved
documents plus 1 exact copy (see below).

| # | Source | doc_id | Text | Reason | sha256 |
|---|---|---|---|---|---|
| 1 | banglishrev | `117135:3` | `valoi to mone hosse` | near_dup (0.875) of BnSentMix test row 14976: `Valoi to mone hossa` | `2b5e8a1b51a0444d0854265dfd7b20557e176588327d9bdbe27594370fcab97e` |
| 2 | banglishrev | `16632:1421` | `eto valo laglo na 💔💔` | near_dup (0.8) of BnSentMix test row 13383: `atto valo laglo na` | `e7f4f51a1356ea7783223aa68b729d83777c7f6a5081804ac99deaa39849873f` |
| 3 | banglishrev | `5474:6` | `price hisab a thik ace,` | near_dup (0.85) of BnSentMix dev row 16188: `Price hisab a thik acha` | `60c2e993297da2c92383a5688a97d0cb2e8ae36c84447526d29b4117d20f9dbe` |
| 4 | banglishrev | `71425:32` | `atto valo laglo na` | exact (1.0) of BnSentMix test row 13383: `atto valo laglo na` | `05b32077844b498c0006d83f2b7cac8428603f9ed9684617b1b5fcdfa65dff62` |
| 5 | banglishrev | `7818:229` | `Masha Allah! / khub valo` | exact (1.0) of BnSentMix test row 16924: `MASHA ALLAH khub valo.......` | `76e4eaa5251e00a50f2253b9433dcc03d65b76311d489f3afb77bd152755a82f` |

All five matched samples are short (3–5 words), formulaic e-commerce phrases (Bucket A). No dev/test sample
of 8+ words was found anywhere in the corpus (Bucket B empty). See docs/prompt2_report.md, Leakage.

## Exact copies removed by content (1 document)

Exclusion matches the **normalized text** of each approved document, not only its doc_id. Reason:
dedup keeps the first copy of a text, so excluding that copy by id let the next identical copy into
the corpus (the first rebuild after removal failed the gate on exactly this). Every copy is listed:

| Source | doc_id | Text | Copy of | sha256 |
|---|---|---|---|---|
| banglishrev | `119136:788` | `masha allah❤️❤️❤️khub valo....` | approved #5 `7818:229` (`Masha Allah! / khub valo`), BnSentMix test row 16924 | `b14fdcaddc4fe48d2a91df04c834ed5623d5b72b3b1c68e684ec8aa3ad988164` |

This copy was not on the list Akib approved; it is the same text after normalization. To revert,
remove the content-based check in `data/corpus.py` (`load_excluded_norm_keys`), and the gate will
fail on test row 16924 again.
