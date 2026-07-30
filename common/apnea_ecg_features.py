"""Stage 1 — Apnea-ECG raw -> structured, per-minute.

Unit of observation: ONE MINUTE of ECG from one record. Labels come from the
database's own `.apn` annotations ('A' = apnoea minute, 'N' = normal minute),
so this is a genuine OSA event-detection task, not a proxy.

Feature basis: the ECG signature of obstructive apnoea is *cyclical variation in
heart rate* (CVHR) — bradycardia during the event, tachycardia on arousal,
repeating on a ~25–100 s cycle (0.01–0.04 Hz) — together with ECG-derived
respiration (EDR), recovered from the beat-to-beat modulation of R-wave
amplitude as the chest moves. Both are computed explicitly below.

Inputs :  dataset_apnea_ecg/raw/*.dat|.hea|.qrs|.apn
Outputs:  dataset_apnea_ecg/structured/minute_features.{parquet,csv}
          dataset_apnea_ecg/structured/subject_level.csv
          dataset_apnea_ecg/structured/data_dictionary.md
          dataset_apnea_ecg/structured/rejects.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, wfdb, glob, os, re, time, json, warnings
from scipy import signal, interpolate

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset_apnea_ecg" / "raw"
OUT = ROOT / "dataset_apnea_ecg" / "structured"
OUT.mkdir(parents=True, exist_ok=True)

FS = 100          # Hz
MIN_SAMPLES = 6000  # 60 s
RR_MIN, RR_MAX = 0.3, 2.5     # physiologically plausible RR bounds (s)
FS_RS = 4.0       # resample rate for the RR tachogram

rejects = []


def clean_rr(t_beats, rr):
    """Drop physiologically implausible intervals and extreme jumps."""
    ok = (rr >= RR_MIN) & (rr <= RR_MAX)
    # remove beats whose RR deviates >30% from the local median (ectopics/misses)
    if ok.sum() > 5:
        med = pd.Series(rr).rolling(11, center=True, min_periods=1).median().to_numpy()
        ok &= np.abs(rr - med) <= 0.3 * med
    return t_beats[ok], rr[ok], int((~ok).sum())


def band_power(f, pxx, lo, hi):
    m = (f >= lo) & (f < hi)
    return float(np.trapezoid(pxx[m], f[m])) if m.sum() > 2 else 0.0


def spectral(t, x, tmin, tmax):
    """Interpolate an irregular series onto a uniform grid and return its PSD."""
    if len(t) < 8:
        return None, None
    grid = np.arange(tmin, tmax, 1.0 / FS_RS)
    if len(grid) < 16:
        return None, None
    f_i = interpolate.interp1d(t, x, kind="linear", bounds_error=False,
                               fill_value=(x[0], x[-1]))
    xs = f_i(grid)
    xs = xs - xs.mean()
    nper = min(len(xs), 256)
    f, pxx = signal.welch(xs, fs=FS_RS, nperseg=nper,
                          noverlap=nper // 2, detrend="linear")
    return f, pxx


def process(rec):
    ann = wfdb.rdann(str(RAW / rec), "apn")
    labels = np.array(ann.symbol)
    starts = ann.sample.astype(np.int64)
    n_min = len(starts)

    qrs = wfdb.rdann(str(RAW / rec), "qrs")
    beats = qrs.sample.astype(np.int64)
    sig = wfdb.rdrecord(str(RAW / rec)).p_signal[:, 0]

    # R-wave amplitude at each detected beat -> EDR carrier
    idx = np.clip(beats, 0, len(sig) - 1)
    ramp = sig[idx]

    tb = beats / FS
    rr = np.diff(tb, prepend=tb[0] - 1.0)
    tb_c, rr_c, n_drop = clean_rr(tb, rr)
    ramp_c = np.interp(tb_c, tb, ramp)
    if n_drop:
        rejects.append(dict(record=rec, reason="implausible_RR_or_ectopic",
                            n_beats_dropped=int(n_drop),
                            pct=round(100 * n_drop / max(len(rr), 1), 2)))

    rows = []
    for i in range(n_min):
        t0 = starts[i] / FS
        t1 = t0 + 60.0
        # 1-minute window for time-domain stats
        m = (tb_c >= t0) & (tb_c < t1)
        r = rr_c[m]
        f = {"record": rec, "minute": i, "label": labels[i]}
        f["n_beats"] = int(m.sum())
        if m.sum() >= 5:
            d = np.diff(r)
            f["rr_mean"] = r.mean()
            f["rr_std"] = r.std()          # SDNN
            f["rr_min"] = r.min()
            f["rr_max"] = r.max()
            f["rr_range"] = r.max() - r.min()
            f["rr_cv"] = r.std() / r.mean() if r.mean() > 0 else np.nan
            f["rmssd"] = np.sqrt((d ** 2).mean()) if len(d) else np.nan
            f["pnn50"] = (np.abs(d) > 0.05).mean() if len(d) else np.nan
            f["hr_mean"] = 60.0 / r.mean() if r.mean() > 0 else np.nan
            f["rr_skew"] = pd.Series(r).skew()
            f["rr_kurt"] = pd.Series(r).kurt()
            f["rr_iqr"] = np.percentile(r, 75) - np.percentile(r, 25)
        else:
            for k in ["rr_mean", "rr_std", "rr_min", "rr_max", "rr_range", "rr_cv",
                      "rmssd", "pnn50", "hr_mean", "rr_skew", "rr_kurt", "rr_iqr"]:
                f[k] = np.nan

        # spectral features over a 5-minute window centred on the minute:
        # CVHR cycles are 25-100 s, so a 1-min window cannot resolve them
        w0, w1 = max(t0 - 120.0, tb_c[0] if len(tb_c) else t0), t1 + 120.0
        ms = (tb_c >= w0) & (tb_c < w1)
        if ms.sum() >= 16:
            fr, px = spectral(tb_c[ms], rr_c[ms], w0, min(w1, tb_c[ms][-1]))
            if fr is not None:
                tot = band_power(fr, px, 0.003, 1.0) + 1e-12
                f["p_cvhr"] = band_power(fr, px, 0.010, 0.040)   # apnoea cycling
                f["p_vlf"] = band_power(fr, px, 0.003, 0.040)
                f["p_lf"] = band_power(fr, px, 0.040, 0.150)
                f["p_hf"] = band_power(fr, px, 0.150, 0.400)
                f["r_cvhr"] = f["p_cvhr"] / tot
                f["r_lf"] = f["p_lf"] / tot
                f["r_hf"] = f["p_hf"] / tot
                f["lf_hf"] = f["p_lf"] / (f["p_hf"] + 1e-12)
                f["p_cvhr_log"] = np.log10(f["p_cvhr"] + 1e-12)
                f["peak_hz"] = float(fr[np.argmax(px)])
                # EDR: respiration recovered from R-wave amplitude modulation
                fe, pe = spectral(tb_c[ms], ramp_c[ms], w0, min(w1, tb_c[ms][-1]))
                if fe is not None:
                    tote = band_power(fe, pe, 0.003, 1.0) + 1e-12
                    f["edr_resp"] = band_power(fe, pe, 0.150, 0.400) / tote
                    f["edr_cvhr"] = band_power(fe, pe, 0.010, 0.040) / tote
                    f["edr_peak_hz"] = float(fe[np.argmax(pe)])
                    f["edr_std"] = float(np.std(ramp_c[ms]))
        rows.append(f)

    df = pd.DataFrame(rows)

    # context windows: apnoea events cluster, so neighbouring minutes inform
    key = ["rr_std", "rr_cv", "rmssd", "p_cvhr_log", "r_cvhr", "lf_hf",
           "hr_mean", "edr_resp"]
    df = df.sort_values("minute").reset_index(drop=True)
    for c in key:
        if c not in df:
            continue
        v = pd.Series(df[c])
        for k in (1, 2, 5):
            df[f"{c}_ctx{k}"] = v.rolling(2 * k + 1, center=True, min_periods=1).mean()
        df[f"{c}_delta"] = v - df[f"{c}_ctx5"]

    # within-record normalisation: removes between-person baseline HR/HRV offsets
    for c in ["rr_mean", "rr_std", "rmssd", "p_cvhr_log", "hr_mean"]:
        if c in df:
            v = df[c].astype(float)
            sd = v.std()
            df[f"{c}_z"] = (v - v.median()) / (sd if sd and sd > 1e-9 else 1.0)
    return df


def main():
    # One record per SUBJECT, and a usable record needs all three of: an ECG
    # channel (.dat), R-peak annotations (.qrs) and minute labels (.apn).
    #
    # This is stricter than "drop anything ending in r" and the difference is
    # load-bearing. Subject c02 has no base record -- the archive ships only
    # c02r, whose four channels are Resp C / Resp A / Resp N / SpO2 with NO ECG,
    # plus c02er.qrs (R-peaks from an ECG that is not distributed here). c02
    # could therefore only be featurised from R-peaks, leaving its EDR columns
    # null. Since c02 is a CONTROL, that would make feature missingness track
    # the label -- a leakage-adjacent artefact worse than the small cohort it
    # would fix. c02 is excluded deliberately, leaving 2 controls (see report).
    def usable(r):
        if not ((RAW / f"{r}.apn").exists() and (RAW / f"{r}.qrs").exists()):
            return False
        try:
            return any("ecg" in s.lower() for s in wfdb.rdheader(str(RAW / r)).sig_name)
        except Exception:
            return False

    avail = {os.path.basename(p)[:-4] for p in glob.glob(str(RAW / "*.dat"))}
    recs, skipped = [], []
    for subj in sorted({re.sub(r"e?r$", "", r) for r in avail}):
        pick = next((c for c in (subj, f"{subj}r", f"{subj}er") if c in avail and usable(c)), None)
        (recs.append(pick) if pick else skipped.append(subj))
    for s in skipped:
        print(f"  {s}: no record with ECG + .qrs + .apn -- excluded")
    print(f"{len(recs)} ECG records with .apn labels ({len(recs)} distinct subjects)")

    frames = []
    for i, r in enumerate(recs, 1):
        t = time.time()
        d = process(r)
        frames.append(d)
        print(f"[{i:2d}/{len(recs)}] {r}: {len(d)} minutes "
              f"(A={int((d.label=='A').sum())}) ({time.time()-t:.1f}s)", flush=True)
    df = pd.concat(frames, ignore_index=True)

    n_in = len(df)
    df["apnea"] = (df["label"] == "A").astype(int)
    df["class_prefix"] = df["record"].str[0]

    # checkpoint: rows in == rows out
    assert len(df) == n_in, "row count changed unexpectedly"

    df.to_parquet(OUT / "minute_features.parquet", index=False)
    df.to_csv(OUT / "minute_features.csv", index=False)
    pd.DataFrame(rejects).to_csv(OUT / "rejects.csv", index=False)

    # ---- subject level: apnoea index from the annotations themselves ----
    g = df.groupby("record")
    sub = pd.DataFrame({
        "n_minutes": g.size(),
        "n_apnea_minutes": g["apnea"].sum(),
    }).reset_index()
    sub["hours"] = sub["n_minutes"] / 60.0
    sub["apnea_index"] = sub["n_apnea_minutes"] / sub["hours"]
    sub["class_prefix"] = sub["record"].str[0]
    sub["osa"] = (sub["apnea_index"] >= 5).astype(int)     # clinical threshold
    sub.to_csv(OUT / "subject_level.csv", index=False)

    print(f"\nsaved {len(df)} minute-rows x {df.shape[1]} cols")
    print("label balance:", df["apnea"].value_counts(normalize=True).round(3).to_dict())
    print("\nsubject level:")
    print(sub[["record", "n_minutes", "apnea_index", "class_prefix", "osa"]]
          .to_string(index=False))
    print("\nOSA (apnea_index>=5):", int(sub.osa.sum()), "/", len(sub))
    print("beats rejected: ", pd.DataFrame(rejects)["n_beats_dropped"].sum()
          if rejects else 0)

    # ---- data dictionary ----
    lines = ["# data_dictionary — minute_features",
             "",
             f"Rows: {len(df)} (one per minute of ECG). "
             f"Records: {df.record.nunique()}. "
             f"Positive class `apnea`=1 share: {df.apnea.mean():.3f}",
             "",
             "| column | dtype | null % | min | max / cardinality |",
             "|---|---|---|---|---|"]
    for c in df.columns:
        s = df[c]
        nullp = f"{s.isna().mean()*100:.1f}"
        if s.dtype.kind in "fi":
            lines.append(f"| {c} | {s.dtype} | {nullp} | {s.min():.4g} | {s.max():.4g} |")
        else:
            lines.append(f"| {c} | {s.dtype} | {nullp} | — | {s.nunique()} distinct |")
    lines += ["", "## Notes",
              "- `label` is the database's own per-minute annotation ('A'/'N'); "
              "`apnea` is its 0/1 encoding and is the modelling target.",
              "- `p_cvhr` / `r_cvhr` capture 0.01–0.04 Hz cyclical variation in "
              "heart rate — the canonical ECG signature of obstructive apnoea.",
              "- `edr_*` are ECG-derived respiration features from R-wave "
              "amplitude modulation.",
              "- `*_ctx{k}` are centred rolling means over ±k minutes; "
              "`*_delta` is the deviation from the ±5-minute mean.",
              "- `*_z` are within-record standardised values.",
              "- Spectral features use a 5-minute window centred on the target "
              "minute, because CVHR cycles (25–100 s) cannot be resolved in 60 s.",
              "",
              "## Leakage note",
              "`class_prefix` and `record` encode the record identity and must "
              "NEVER be used as model features. `n_apnea_minutes` / "
              "`apnea_index` are derived from the labels and are subject-level "
              "outcome summaries, not predictors."]
    (OUT / "data_dictionary.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote data_dictionary.md")


if __name__ == "__main__":
    main()
