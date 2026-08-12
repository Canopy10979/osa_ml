# ucddb_v2 — raw source

**UCD Sleep Apnea Database (UCDDB)**, St. Vincent's University Hospital, Dublin.
25 full overnight polysomnograms, 1.3 GB. Not tracked in git — see `.gitignore`.

Distinct from the earlier `dataset_ucddb/` folder (now removed), which held only the
12 KB `SubjectDetails.xls` demographics table with no recordings.

## Layout

| Path | Contents |
|---|---|
| `ucddbNNN.rec` | 14-channel PSG, EDF format — SpO2 (8 Hz), Flow (8 Hz), ribcage/abdo (8 Hz), Sum, Sound, BodyPos, Pulse, ECG (128 Hz), EEG C3A2/C4A1 (128 Hz), EOG, EMG |
| `ucddbNNN_lifecard.edf` | 3-channel Holter ECG, 128 Hz (not used by the current pipeline) |
| `ucddbNNN_respevt.txt` | expert respiratory event list — wall-clock start time, type, duration, SpO2 low/%drop, snore, arousal |
| `ucddbNNN_stage.txt` | one sleep-stage code per 30 s epoch (0 = wake, 8 = artefact) |
| `SubjectDetails.xls` | PSG AHI, BMI, age, gender, Epworth score, study duration, sleep efficiency |
| `RECORDS`, `SHA256SUMS.txt` | manifest and checksums |

## Cohort

25 subjects, **PSG AHI 2–91, median 16**. At AHI ≥ 15 the split is **14 OSA / 11
non-OSA**. Notably **16 of 25 sit in the AHI 5–25 band** — the diagnostically ambiguous
range absent from `dataset_apnea_hrv`.

## Note on time alignment

Events are stamped in wall-clock time while epochs index from PSG start, and recordings
cross midnight. `common/ucddb_features.py` aligns using the **EDF header start time**
(written by the recorder, and what the sample indices are relative to) rather than the
spreadsheet's `PSG Start Time`. The alignment is validated by requiring the derived event
index to reproduce the recorded PSG AHI: **r = 0.978**.
