"""Stages 1-2 for dataset_apnea_hrv (HuGCDN2014): raw .mat -> structured features.

77 single-lead ECG recordings, expert-scored for apnoea in every minute from
simultaneous PSG. 40 controls (AHI < 5) and 37 patients (AHI > 25) -- the only
dataset in this repo with a large, balanced, well-separated OSA/non-OSA cohort.

The archive ships RR interval series already framed: one 5-minute frame per
minute, shifted in 1-minute increments, the score belonging to the centre
minute. So one row here = one minute, featurised from the 5 minutes around it.

Features target the two established ECG signatures of obstructive apnoea:
  * cyclical variation in heart rate (CVHR) -- the ~0.01-0.04 Hz
    bradycardia/tachycardia cycling driven by repetitive apnoea/arousal
  * respiratory sinus arrhythmia suppression -- HF (0.15-0.40 Hz) power falls
    when breathing stops

Outputs:
  structured/minute_features.{parquet,csv}   one row per subject-minute
  structured/subject_level.csv               one row per subject
  structured/data_dictionary.md              column, dtype, units, null %, range
  structured/rejects.csv                     per-subject beat QC drops
  results/leakage_audit.csv                  univariate AUC screen
"""
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio, time, sys
from scipy import signal, stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset_apnea_hrv"
RAW, S, R = DS / "raw", DS / "structured", DS / "results"
for d in (S, R):
    d.mkdir(parents=True, exist_ok=True)

FS = 4.0                      # resample rate for spectral analysis (Hz)
RR_MIN, RR_MAX = 300.0, 2000.0   # physiologically plausible RR (ms)
ECTOPIC = 0.20                # reject beat if RR jumps >20% vs previous

BANDS = {
    "cvhr": (0.010, 0.040),   # apnoea/arousal cycling -- the mechanism of interest
    "lf":   (0.040, 0.150),   # baroreflex / sympathetic
    "hf":   (0.150, 0.400),   # respiratory sinus arrhythmia
}

rejects = []


def clean_rr(rr, subj):
    """Drop implausible and ectopic beats. Returns cleaned array + n dropped."""
    rr = np.asarray(rr, float).ravel()
    n0 = rr.size
    rr = rr[(rr >= RR_MIN) & (rr <= RR_MAX)]
    if rr.size > 2:
        d = np.abs(np.diff(rr)) / rr[:-1]
        rr = rr[np.r_[True, d <= ECTOPIC]]
    return rr, n0 - rr.size


def spectral(rr):
    """Welch PSD of the RR tachogram, interpolated onto an even 4 Hz grid."""
    t = np.cumsum(rr) / 1000.0
    if t.size < 16 or (t[-1] - t[0]) < 30:
        return None
    grid = np.arange(t[0], t[-1], 1 / FS)
    x = np.interp(grid, t, rr)
    x = signal.detrend(x)
    nper = min(len(x), int(120 * FS))       # 120 s segments
    f, p = signal.welch(x, fs=FS, nperseg=nper)
    return f, p


def frame_features(rr):
    """HRV features for one 5-minute frame. Returns dict or None if unusable."""
    if rr.size < 30:
        return None
    d = np.diff(rr)
    f = dict(
        n_beats=rr.size,
        rr_mean=rr.mean(), rr_median=np.median(rr), rr_std=rr.std(ddof=1),
        rr_cv=rr.std(ddof=1) / rr.mean(),
        rr_min=rr.min(), rr_max=rr.max(), rr_range=np.ptp(rr),
        rr_iqr=np.subtract(*np.percentile(rr, [75, 25])),
        rr_mad=np.median(np.abs(rr - np.median(rr))),
        rr_skew=stats.skew(rr), rr_kurt=stats.kurtosis(rr),
        hr_mean=60000.0 / rr.mean(),
        rmssd=np.sqrt((d ** 2).mean()) if d.size else np.nan,
        sdsd=d.std(ddof=1) if d.size > 1 else np.nan,
        pnn50=(np.abs(d) > 50).mean() if d.size else np.nan,
        pnn20=(np.abs(d) > 20).mean() if d.size else np.nan,
    )
    # Poincare descriptors
    if d.size > 1:
        f["sd1"] = np.sqrt(0.5) * d.std(ddof=1)
        f["sd2"] = np.sqrt(max(2 * rr.var(ddof=1) - 0.5 * d.var(ddof=1), 0))
        f["sd_ratio"] = f["sd1"] / f["sd2"] if f["sd2"] > 0 else np.nan

    sp = spectral(rr)
    if sp is None:
        for b in BANDS:
            f[f"p_{b}"] = f[f"r_{b}"] = np.nan
        f["lf_hf"] = f["peak_hz"] = f["cvhr_peak_hz"] = f["total_power"] = np.nan
        return f

    fr, ps = sp
    tot = np.trapezoid(ps[(fr >= 0.003) & (fr <= 0.4)], fr[(fr >= 0.003) & (fr <= 0.4)])
    f["total_power"] = tot
    for b, (lo, hi) in BANDS.items():
        m = (fr >= lo) & (fr < hi)
        p = np.trapezoid(ps[m], fr[m]) if m.sum() > 1 else np.nan
        f[f"p_{b}"] = p
        f[f"r_{b}"] = p / tot if tot and tot > 0 else np.nan
    f["lf_hf"] = f["p_lf"] / f["p_hf"] if f.get("p_hf") else np.nan
    f["peak_hz"] = fr[np.argmax(ps)] if ps.size else np.nan
    mc = (fr >= BANDS["cvhr"][0]) & (fr < BANDS["cvhr"][1])
    f["cvhr_peak_hz"] = fr[mc][np.argmax(ps[mc])] if mc.sum() > 1 else np.nan
    return f


def add_context(df, cols, lags=(1, 2, 5)):
    """Neighbouring-minute context. Apnoea is temporally clustered, so what the
    surrounding minutes look like is genuinely informative -- and it uses only
    features, never labels, so it introduces no leakage."""
    out = [df]
    g = df.groupby("subject", sort=False)
    for L in lags:
        roll = g[cols].transform(lambda s: s.rolling(2 * L + 1, center=True, min_periods=1).mean())
        out.append(roll.add_suffix(f"_ctx{L}"))
    return pd.concat(out, axis=1)


def main():
    subs = sorted(p.stem for p in (RAW / "RR").glob("*.mat"))
    print(f"{len(subs)} subjects")
    frames = []
    t0 = time.time()

    for i, s in enumerate(subs, 1):
        cells = sio.loadmat(RAW / "RR" / f"{s}.mat")["RR_notch_abs_pr_ada"].ravel()
        lab = sio.loadmat(RAW / "LABELS" / f"{s}.mat")["salida_man_1m"].ravel()
        n = min(len(cells), len(lab))       # RR frames and labels differ by <=1
        dropped = kept = 0
        rows = []
        for m in range(n):
            rr, nd = clean_rr(cells[m], s)
            dropped += nd
            fe = frame_features(rr)
            if fe is None:
                continue
            fe.update(subject=s, minute=m, apnea=int(lab[m]))
            rows.append(fe)
            kept += 1
        rejects.append(dict(subject=s, n_minutes=n, minutes_kept=kept,
                            minutes_dropped=n - kept,
                            reason="empty_or_too_few_beats" if n - kept else "",
                            beats_dropped=dropped))
        frames.append(pd.DataFrame(rows))
        if i % 10 == 0 or i == len(subs):
            print(f"  [{i:2d}/{len(subs)}] {s}  {time.time()-t0:.0f}s")

    df = pd.concat(frames, ignore_index=True)

    base = [c for c in df.columns if c not in ("subject", "minute", "apnea")]
    ctx_src = ["rr_std", "rr_cv", "rmssd", "lf_hf", "r_hf", "r_cvhr", "hr_mean"]
    df = add_context(df, [c for c in ctx_src if c in df.columns])
    df = df.replace([np.inf, -np.inf], np.nan)

    # ---- subject level ----
    sub = (df.groupby("subject")
             .agg(n_minutes=("apnea", "size"), n_apnea=("apnea", "sum"))
             .reset_index())
    sub["hours"] = sub.n_minutes / 60.0
    sub["apnea_index"] = sub.n_apnea / sub.hours

    # Group assignment comes from the record-name prefix -- APNxxx = patient
    # (AHI > 25), CONxxx = control (AHI < 5). That is the archive's own
    # convention and needs no inference.
    #
    # It is cross-checked against the annotated apnoea index, which under the
    # documented AHI<5 / AHI>25 design must be cleanly separable: every control
    # below every patient, with no overlap. If that check ever fails, either the
    # labels or the file naming is wrong and the run should not proceed.
    sub["group"] = sub.subject.str[:3].map({"APN": "APNEA", "CON": "CONTROL"})
    sub["osa"] = (sub.group == "APNEA").astype(int)
    con_max = sub.loc[sub.group == "CONTROL", "apnea_index"].max()
    apn_min = sub.loc[sub.group == "APNEA", "apnea_index"].min()

    # official split: first 20 controls + first 18 patients (by record number)
    sub = sub.sort_values("subject").reset_index(drop=True)
    learn = pd.concat([sub[sub.group == "CONTROL"].head(20),
                       sub[sub.group == "APNEA"].head(18)]).subject
    sub["split"] = np.where(sub.subject.isin(learn), "L", "T")

    df = df.merge(sub[["subject", "osa", "group", "split"]], on="subject", how="left")

    # ---- validation ----
    ok = True
    def chk(name, val, passed):
        nonlocal ok
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {val}")

    print("\n-- validation --")
    chk("subjects", df.subject.nunique(), df.subject.nunique() == 77)
    chk("every record named APN*/CON*", int(sub.group.notna().sum()), sub.group.notna().all())
    chk("controls == 40", int((sub.group == "CONTROL").sum()), (sub.group == "CONTROL").sum() == 40)
    chk("patients == 37", int((sub.group == "APNEA").sum()), (sub.group == "APNEA").sum() == 37)
    chk("groups separate cleanly on annotated index",
        f"control max {con_max:.2f} < patient min {apn_min:.2f}", con_max < apn_min)
    chk("learning set == 38", int((sub.split == "L").sum()), (sub.split == "L").sum() == 38)
    chk("labels binary", sorted(df.apnea.unique()), set(df.apnea.unique()) <= {0, 1})
    chk("no all-null feature", int(df[base].isna().all().sum()), df[base].isna().all().sum() == 0)
    chk("minutes retained >= 95%",
        f"{df.shape[0]}/{sum(r['n_minutes'] for r in rejects)}",
        df.shape[0] >= 0.95 * sum(r["n_minutes"] for r in rejects))

    print(f"\n  minutes: {len(df):,}   apnoea: {df.apnea.mean():.1%}")
    print(f"  subjects: 40 control / 37 apnea; split L={int((sub.split=='L').sum())} T={int((sub.split=='T').sum())}")

    # ---- Stage 2 leakage audit ----
    feats = [c for c in df.columns
             if c not in ("subject", "minute", "apnea", "osa", "group", "split")
             and df[c].dtype.kind in "fi"]
    aud = []
    for c in feats:
        v = df[c]
        m = v.notna()
        if m.sum() < 100 or v[m].nunique() < 2:
            continue
        a = roc_auc_score(df.apnea[m], v[m])
        aud.append(dict(feature=c, univariate_auc=max(a, 1 - a),
                        point_biserial_r=stats.pointbiserialr(df.apnea[m], v[m])[0],
                        null_pct=round(100 * (1 - m.mean()), 3),
                        flag="SUSPECT >0.90" if max(a, 1 - a) > 0.90 else ""))
    aud = pd.DataFrame(aud).sort_values("univariate_auc", ascending=False)
    aud.to_csv(R / "leakage_audit.csv", index=False)
    nsus = int((aud.univariate_auc > 0.90).sum())
    chk("no single-feature separator (AUC>0.90)", nsus, nsus == 0)
    print(f"  strongest single feature: {aud.iloc[0].feature} AUC={aud.iloc[0].univariate_auc:.3f}")

    # ---- write ----
    df.to_parquet(S / "minute_features.parquet", index=False)
    df.to_csv(S / "minute_features.csv", index=False)
    sub.to_csv(S / "subject_level.csv", index=False)
    pd.DataFrame(rejects).to_csv(S / "rejects.csv", index=False)

    lines = ["# data_dictionary — dataset_apnea_hrv", "",
             f"`minute_features.parquet` — {len(df):,} rows x {df.shape[1]} cols. "
             "One row = one minute, featurised from the 5-minute frame centred on it.", "",
             "| column | dtype | null % | min | max |", "|---|---|---|---|---|"]
    for c in df.columns:
        v = df[c]
        if v.dtype.kind in "fi":
            lines.append(f"| `{c}` | {v.dtype} | {100*v.isna().mean():.2f} | {v.min():.4g} | {v.max():.4g} |")
        else:
            lines.append(f"| `{c}` | {v.dtype} | {100*v.isna().mean():.2f} | — | {v.nunique()} distinct |")
    lines += ["", "## Units", "",
              "RR intervals in ms; `hr_mean` in bpm; `p_*` band powers in ms²/Hz; "
              "`r_*` relative (fraction of total power); `*_ctx{n}` = centred rolling "
              "mean over ±n minutes within subject.", "",
              "`apnea` = expert per-minute label (1 = apnoea). `osa` = subject-level "
              "group (1 = APNEA cohort, AHI > 25). `split` = official L/T sets."]
    (S / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n[write] {S/'minute_features.parquet'}  ({len(df):,} x {df.shape[1]})")
    print(f"[write] {S/'subject_level.csv'}, rejects.csv, data_dictionary.md")
    print(f"[write] {R/'leakage_audit.csv'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
