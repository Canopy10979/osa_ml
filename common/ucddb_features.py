"""Stages 1-2 for dataset_ucddb_v2 (UCD Sleep Apnea Database): raw PSG -> features.

25 full overnight polysomnograms from St. Vincent's University Hospital, Dublin.
Unlike the other datasets in this repo, this one carries the channels clinicians
actually score apnoea from -- **SpO2, nasal flow, and ribcage/abdomen effort** --
not just ECG. It is also the only cohort spanning the diagnostically ambiguous
AHI 5-25 range.

  .rec              14-channel PSG (SpO2 8 Hz, Flow 8 Hz, ribcage/abdo 8 Hz, ECG 128 Hz)
  _respevt.txt      expert respiratory event list (clock time, type, duration)
  _stage.txt        one sleep-stage code per 30 s epoch
  SubjectDetails.xls  PSG AHI, demographics

Unit of observation = one 30 s epoch, labelled 1 if an expert-scored respiratory
event overlaps it.

The load-bearing risk here is TIME ALIGNMENT: events are stamped in wall-clock
time while epochs are indexed from PSG start, so a wrong offset would silently
scramble every label. The validation gate is that the derived event index must
correlate with the independently-recorded PSG AHI -- if alignment were wrong,
that correlation would collapse.

Outputs:
  structured/epoch_features.{parquet,csv}   one row per subject-epoch
  structured/subject_level.csv              one row per subject
  structured/data_dictionary.md
  structured/rejects.csv
  results/leakage_audit.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, re, sys, json, warnings
from scipy import stats, signal as sig
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset_ucddb_v2"
RAW, S, R = DS / "raw", DS / "structured", DS / "results"
for d in (S, R):
    d.mkdir(parents=True, exist_ok=True)

EPOCH = 30.0          # seconds per scored epoch
WANT = ["SpO2", "Flow", "ribcage", "abdo", "Sum", "Pulse", "Sound"]

# Obstructive/central/mixed apnoeas and hypopnoeas all count as respiratory
# events. PB (periodic breathing) and POSSIBLE are excluded: PB is a breathing
# *pattern* spanning minutes rather than a discrete event, and POSSIBLE is by
# definition uncertain -- including either would blur the label.
EVENT_TYPES = {"APNEA-O", "APNEA-C", "APNEA-M", "HYP-O", "HYP-C", "HYP-M"}

rejects = []


def read_edf(path, want):
    """Minimal EDF reader supporting per-signal sample rates.

    Returns {label: (signal, fs)} for the requested labels, plus the header
    start time. Written out rather than taking an MNE dependency for one read.
    """
    with open(path, "rb") as f:
        f.read(8); f.read(80); f.read(80)
        f.read(8)                                     # start date
        start = f.read(8).decode("ascii", "replace")  # hh.mm.ss
        nb_hdr = int(f.read(8)); f.read(44)
        nrec = int(f.read(8)); dur = float(f.read(8)); ns = int(f.read(4))
        lab = [f.read(16).decode("ascii", "replace").strip() for _ in range(ns)]
        f.read(80 * ns)
        f.read(8 * ns)                                # units
        pmin = np.array([float(f.read(8)) for _ in range(ns)])
        pmax = np.array([float(f.read(8)) for _ in range(ns)])
        dmin = np.array([float(f.read(8)) for _ in range(ns)])
        dmax = np.array([float(f.read(8)) for _ in range(ns)])
        f.read(80 * ns)
        nsamp = np.array([int(f.read(8)) for _ in range(ns)])
        f.seek(nb_hdr)
        raw = np.fromfile(f, dtype="<i2")

    per_rec = int(nsamp.sum())
    nrec = min(nrec, len(raw) // per_rec)
    raw = raw[: nrec * per_rec].reshape(nrec, per_rec)

    span = np.r_[0, np.cumsum(nsamp)]
    scale = (pmax - pmin) / np.where(dmax - dmin == 0, 1, dmax - dmin)
    out = {}
    for i, name in enumerate(lab):
        if name not in want:
            continue
        x = raw[:, span[i]:span[i + 1]].astype(np.float32).ravel()
        out[name] = ((x - dmin[i]) * scale[i] + pmin[i], nsamp[i] / dur)
    return out, start.replace(".", ":"), nrec * dur


def parse_events(path):
    """(start_seconds_since_midnight, duration_s, type) for each scored event."""
    ev = []
    for ln in Path(path).read_text(errors="replace").splitlines()[3:]:
        m = re.match(r"\s*(\d{1,2}):(\d{2}):(\d{2})\s+(\S+)\s+(.*)$", ln)
        if not m:
            continue
        h, mi, s, typ, rest = m.groups()
        if typ not in EVENT_TYPES:
            continue
        d = re.match(r"\s*(?:\S+\s+)??(\d+)\s", rest)      # duration column
        dur = float(d.group(1)) if d else 15.0
        ev.append((int(h) * 3600 + int(mi) * 60 + int(s), dur, typ))
    return ev


def to_seconds(hms):
    h, m, s = [int(float(x)) for x in hms.split(":")]
    return h * 3600 + m * 60 + s


def epoch_labels(events, t0, n_ep):
    """Mark an epoch positive if any scored event overlaps it.

    Events carry wall-clock times and the recording may cross midnight, so each
    event is unwrapped to seconds-since-recording-start before binning.
    """
    y = np.zeros(n_ep, dtype=int)
    for t_abs, dur, _ in events:
        rel = t_abs - t0
        if rel < -12 * 3600:      # event after midnight, recording began before
            rel += 24 * 3600
        elif rel > 12 * 3600:     # guard the opposite wrap
            rel -= 24 * 3600
        if rel < -dur:
            continue
        a = max(int(np.floor(rel / EPOCH)), 0)
        b = min(int(np.ceil((rel + dur) / EPOCH)), n_ep)
        if a < n_ep:
            y[a:max(b, a + 1)] = 1
    return y


def win_feats(x, fs, n_ep, prefix):
    """Per-epoch summaries of one channel, plus a 2-minute rolling baseline."""
    per = int(round(EPOCH * fs))
    usable = min(n_ep, len(x) // per) if per else 0
    X = x[: usable * per].reshape(usable, per).astype(float)
    f = {}
    f[f"{prefix}_mean"] = X.mean(1)
    f[f"{prefix}_std"] = X.std(1)
    f[f"{prefix}_min"] = X.min(1)
    f[f"{prefix}_max"] = X.max(1)
    f[f"{prefix}_range"] = X.max(1) - X.min(1)
    f[f"{prefix}_iqr"] = np.subtract(*np.percentile(X, [75, 25], axis=1))
    # amplitude relative to a 2-minute trailing baseline: a flow or effort
    # reduction is only meaningful against what that subject was doing before
    base = pd.Series(f[f"{prefix}_std"]).rolling(4, min_periods=1).max()
    f[f"{prefix}_rel_amp"] = f[f"{prefix}_std"] / base.replace(0, np.nan).to_numpy()
    out = pd.DataFrame(f)
    return out.reindex(range(n_ep))


def process(subj, meta):
    rec = RAW / f"{subj}.rec"
    if not rec.exists():
        rejects.append(dict(subject=subj, reason="no .rec file")); return None
    sigs, edf_start, dur_s = read_edf(rec, WANT)
    stages = np.array([int(v) for v in
                       (RAW / f"{subj}_stage.txt").read_text().split() if v.strip()])
    n_ep = min(len(stages), int(dur_s // EPOCH))
    if n_ep < 60:
        rejects.append(dict(subject=subj, reason=f"too few epochs ({n_ep})")); return None

    # Prefer the EDF header start time over the spreadsheet: it is written by the
    # recorder itself and is what the signal indices are actually relative to.
    t0 = to_seconds(edf_start)
    y = epoch_labels(parse_events(RAW / f"{subj}_respevt.txt"), t0, n_ep)

    cols = [win_feats(v[0], v[1], n_ep, k.lower()) for k, v in sigs.items()]
    df = pd.concat(cols, axis=1).iloc[:n_ep].reset_index(drop=True)

    # SpO2 desaturation depth against a 2-min rolling baseline -- the single
    # most clinically direct marker of an apnoeic event.
    if "spo2_mean" in df:
        b = df.spo2_mean.rolling(4, min_periods=1).max()
        df["spo2_desat"] = b - df.spo2_min
        df["spo2_below90"] = (df.spo2_min < 90).astype(float)
        df["spo2_below92"] = (df.spo2_min < 92).astype(float)
    if {"ribcage_std", "abdo_std"} <= set(df.columns):
        # thoraco-abdominal asynchrony: effort continues but out of phase during
        # obstructive events
        df["effort_ratio"] = df.ribcage_std / df.abdo_std.replace(0, np.nan)
        df["effort_sum"] = df.ribcage_std + df.abdo_std

    df.insert(0, "subject", subj)
    df.insert(1, "epoch", np.arange(n_ep))
    df["stage"] = stages[:n_ep]
    df["asleep"] = (~np.isin(stages[:n_ep], [0, 8])).astype(int)
    df["event"] = y

    ctx = [c for c in ("spo2_desat", "spo2_min", "flow_rel_amp", "effort_sum",
                       "pulse_mean", "pulse_std") if c in df.columns]
    for L in (1, 2):
        roll = df[ctx].rolling(2 * L + 1, center=True, min_periods=1).mean()
        df = pd.concat([df, roll.add_suffix(f"_ctx{L}")], axis=1)
    return df


def main():
    meta = pd.read_excel(RAW / "SubjectDetails.xls")
    meta["subject"] = meta["Study Number"].str.strip().str.lower()
    subs = sorted(p.stem for p in RAW.glob("ucddb*.rec"))
    print(f"{len(subs)} PSG recordings")

    frames = []
    for i, s in enumerate(subs, 1):
        d = process(s, meta)
        if d is not None:
            frames.append(d)
            print(f"  [{i:2d}/{len(subs)}] {s}: {len(d)} epochs, "
                  f"{d.event.mean():.1%} event, {d.asleep.mean():.0%} asleep")
    df = pd.concat(frames, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    # ---- subject level ----
    sub = (df.groupby("subject")
             .agg(n_epochs=("event", "size"), n_event_epochs=("event", "sum"),
                  sleep_epochs=("asleep", "sum")).reset_index())
    sub["sleep_hours"] = sub.sleep_epochs * EPOCH / 3600
    sub["event_index"] = sub.n_event_epochs / sub.sleep_hours.replace(0, np.nan)
    sub = sub.merge(meta[["subject", "PSG AHI", "BMI", "Age", "Gender",
                          "Epworth Sleepiness Score"]], on="subject", how="left")
    sub = sub.rename(columns={"PSG AHI": "ahi", "Epworth Sleepiness Score": "epworth"})
    sub["osa_15"] = (sub.ahi >= 15).astype(int)
    sub["osa_5"] = (sub.ahi >= 5).astype(int)
    df = df.merge(sub[["subject", "ahi", "osa_15"]], on="subject", how="left")

    # ---- validation ----
    ok = True
    def chk(name, val, passed):
        nonlocal ok; ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {val}")

    print("\n-- validation --")
    chk("subjects", df.subject.nunique(), df.subject.nunique() == 25)
    chk("all subjects matched to metadata", int(sub.ahi.notna().sum()), sub.ahi.notna().all())
    chk("labels binary", sorted(df.event.unique()), set(df.event.unique()) <= {0, 1})
    chk("every subject has >=1 event", int((sub.n_event_epochs > 0).sum()),
        (sub.n_event_epochs > 0).all())

    # THE alignment gate.
    r = stats.pearsonr(sub.event_index, sub.ahi)
    chk("derived event index tracks recorded AHI (r>0.7)",
        f"r={r[0]:.3f} p={r[1]:.2g}", r[0] > 0.7)

    chk("events concentrated in sleep",
        f"{df[df.asleep==1].event.mean():.1%} asleep vs {df[df.asleep==0].event.mean():.1%} awake",
        df[df.asleep == 1].event.mean() > df[df.asleep == 0].event.mean())

    print(f"\n  epochs: {len(df):,}   event epochs: {df.event.mean():.1%}")
    print(f"  OSA at AHI>=15: {int(sub.osa_15.sum())} / {len(sub)}  "
          f"(AHI>=5: {int(sub.osa_5.sum())})")

    # ---- leakage audit ----
    DROP = {"subject", "epoch", "event", "ahi", "osa_15", "stage"}
    feats = [c for c in df.columns if c not in DROP and df[c].dtype.kind in "fi"]
    aud = []
    for c in feats:
        v = df[c]; m = v.notna()
        if m.sum() < 200 or v[m].nunique() < 2:
            continue
        a = roc_auc_score(df.event[m], v[m])
        aud.append(dict(feature=c, univariate_auc=max(a, 1 - a),
                        point_biserial_r=stats.pointbiserialr(df.event[m], v[m])[0],
                        null_pct=round(100 * (1 - m.mean()), 3),
                        flag="SUSPECT >0.90" if max(a, 1 - a) > 0.90 else ""))
    aud = pd.DataFrame(aud).sort_values("univariate_auc", ascending=False)
    aud.to_csv(R / "leakage_audit.csv", index=False)
    chk("no single-feature separator (AUC>0.90)",
        int((aud.univariate_auc > 0.90).sum()), (aud.univariate_auc <= 0.90).all())
    print(f"  strongest single feature: {aud.iloc[0].feature} AUC={aud.iloc[0].univariate_auc:.3f}")

    df.to_parquet(S / "epoch_features.parquet", index=False)
    df.to_csv(S / "epoch_features.csv", index=False)
    sub.to_csv(S / "subject_level.csv", index=False)
    pd.DataFrame(rejects or [{"subject": "-", "reason": "none"}]).to_csv(
        S / "rejects.csv", index=False)

    lines = ["# data_dictionary — dataset_ucddb_v2", "",
             f"`epoch_features.parquet` — {len(df):,} rows x {df.shape[1]} cols. "
             "One row = one 30 s epoch.", "",
             "| column | dtype | null % | min | max |", "|---|---|---|---|---|"]
    for c in df.columns:
        v = df[c]
        if v.dtype.kind in "fi":
            lines.append(f"| `{c}` | {v.dtype} | {100*v.isna().mean():.2f} | {v.min():.4g} | {v.max():.4g} |")
        else:
            lines.append(f"| `{c}` | {v.dtype} | {100*v.isna().mean():.2f} | — | {v.nunique()} distinct |")
    lines += ["", "## Notes", "",
              "`event` = 1 if an expert-scored apnoea/hypopnoea (obstructive, central or "
              "mixed) overlaps the epoch. PB and POSSIBLE are excluded.",
              "`asleep` = stage not in {0 wake, 8 artefact}. `ahi` is the recorded PSG AHI; "
              "`osa_15` = AHI >= 15. Both are subject-level and must never be used as features.",
              "`*_rel_amp` = epoch amplitude over a 2-minute rolling maximum. "
              "`spo2_desat` = 2-minute baseline SpO2 minus epoch minimum."]
    (S / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[write] {S/'epoch_features.parquet'} ({len(df):,} x {df.shape[1]})")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
