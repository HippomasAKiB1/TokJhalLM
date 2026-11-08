# NSFW filter D16 replay (redacted)

The Prompt 2 spot-check sample, reproduced exactly with the pre-D16 rule (same seed, same reservoir order, verified row by row against the local review file by URL), re-scored with the D16 rule. Verdicts are the Prompt 2 verdicts confirmed by Akib. URLs and text are omitted; the local file docs/nsfw_spotcheck.md has them.

## fineweb2_ben_latn

Sample reproduced: yes, all 20 rows match. Docs checked: 10,581. Old rule drops: 4,202. New rule drops: 3,883. Old-dropped now kept: 320. Old-kept now dropped: 1.

| # | old rule fired | new rule | Prompt 2 verdict | outcome |
|---|---|---|---|---|
| 1 | terms:choda,chodachudi,chodar,choti,chuda,chude,chudlam,khanki,magi,sex,sexy | terms | TP | TP still dropped |
| 2 | url:choti,chod | url:chod,choti | TP | TP still dropped |
| 3 | terms:choda,chodachudi,chodar,choti,chuda,chude,chudte,magi | terms | TP | TP still dropped |
| 4 | terms:choda,choti,chuda,chude,chudte,khanki,magi,sex | terms | BORDERLINE | borderline, still dropped |
| 5 | url:choti | url:choti | TP | TP still dropped |
| 6 | url:chud,chuda | url:chud | TP | TP still dropped |
| 7 | url:chud | url:chud | FP | FP STILL DROPPED |
| 8 | terms:choda,chodar,choti,chuda,chude,chudlam,chudte,sex,sexy | terms | TP | TP still dropped |
| 9 | url:sex | kept | TP | TP REGRESSION (now kept) |
| 10 | url:chod,chud | url:chod,chud | TP | TP still dropped |
| 11 | url:chod | url:chod | TP | TP still dropped |
| 12 | terms:choda,chodachudi,chodar,choti,chuda,porn | terms | TP | TP still dropped |
| 13 | terms:choda,chodachudi,chodar,choti,chuda,chudlam,khanki | terms | TP | TP still dropped |
| 14 | terms:choda,chodachudi,chodar,choti,chuda,chude,chudlam,magi,sex,sexy | terms | TP | TP still dropped |
| 15 | url:choti | url:choti | TP | TP still dropped |
| 16 | url:chud | url:chud | TP | TP still dropped |
| 17 | url:choti,chod | url:chod | TP | TP still dropped |
| 18 | url:adult | kept | BORDERLINE | borderline, now kept |
| 19 | url:chud | url:chud | TP | TP still dropped |
| 20 | url:chod,chud | url:chod,chud | TP | TP still dropped |

Tally: FP STILL DROPPED: 1, TP REGRESSION (now kept): 1, TP still dropped: 16, borderline, now kept: 1, borderline, still dropped: 1

## fineweb2_ben_beng

Sample reproduced: yes, all 20 rows match. Docs checked: 322,000. Old rule drops: 1,088. New rule drops: 770. Old-dropped now kept: 467. Old-kept now dropped: 149.

| # | old rule fired | new rule | Prompt 2 verdict | outcome |
|---|---|---|---|---|
| 1 | url:porn | url:porn | TP | TP still dropped |
| 2 | url:choti | url:choti | TP | TP still dropped |
| 3 | url:porn | url:porn | TP | TP still dropped |
| 4 | url:sex | kept | FP | FP recovered (kept) |
| 5 | url:sex | kept | FP | FP recovered (kept) |
| 6 | url:porn | url:porn | TP | TP still dropped |
| 7 | url:porn | url:porn | TP | TP still dropped |
| 8 | url:choti,sex | url:choti | TP | TP still dropped |
| 9 | url:sex | kept | FP | FP recovered (kept) |
| 10 | url:choti | url:choti | TP | TP still dropped |
| 11 | url:porn | url:porn | TP | TP still dropped |
| 12 | url:sex | kept | FP | FP recovered (kept) |
| 13 | url:porn | url:porn | TP | TP still dropped |
| 14 | url:choti | url:choti | TP | TP still dropped |
| 15 | url:sex | kept | FP | FP recovered (kept) |
| 16 | url:porn | url:porn | TP | TP still dropped |
| 17 | url:choti,chod | url:chod,choti | TP | TP still dropped |
| 18 | url:chud | kept | FP | FP recovered (kept) |
| 19 | url:sex | kept | FP | FP recovered (kept) |
| 20 | url:sex | kept | BORDERLINE | borderline, now kept |

Tally: FP recovered (kept): 7, TP still dropped: 12, borderline, now kept: 1

