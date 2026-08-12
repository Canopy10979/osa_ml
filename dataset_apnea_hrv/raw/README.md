# apnea_hrv — raw source

**HuGCDN2014 database**, provided by the sleep unit of Dr. Negrín University
Hospital (Canary Islands, Spain). 46 MB. Not tracked in git — see `.gitignore`.

77 single-lead ECG recordings digitized at 200 Hz. An expert scored the presence
or absence of apnoea in **each minute** from simultaneous polysomnography.

| Group | n | Criterion |
|---|---|---|
| CONTROL | 40 | AHI < 5 (30 men, 10 women) |
| APNEA | 37 | AHI > 25 (30 men, 7 women) |

The ECG is divided into **5-minute frames shifted in 1-minute increments**; each
frame's score is assigned to the minute at its centre. The RR interval series is
the sequence of time differences between successive heartbeats.

## Layout

| Path | Contents |
|---|---|
| `RR/APNxxx.mat` | key `RR_notch_abs_pr_ada` — 1×N cell array, one 5-min frame per minute, RR intervals in ms |
| `LABELS/APNxxx.mat` | key `salida_man_1m` — 1×N per-minute labels (0/1); `salida_man` is the 30-s resolution version |
| `Readme.docx` | original database description |

## Note on group assignment

The archive does **not** record which `APNxxx` files are controls and which are
patients. The split is derived in `common/apnea_hrv_features.py` from each
subject's annotated apnoea-minute index, which is sharply bimodal: 40 subjects
fall at or below 3.5 events/h and 37 at or above 16.8, with **nothing in
between**. That reproduces the documented 40/37 split exactly, which is the
check that the derivation is correct.

The official learning set (first 20 controls + first 18 patients, by record
number) is reconstructed the same way and used as the held-out split.
