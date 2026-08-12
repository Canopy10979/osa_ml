"""cross_dataset — corrected analysis: apnoea events, not sleep/wake.

WHAT WAS WRONG
--------------
The legacy outputs in `cross_dataset/results/` reported up to 97% accuracy for
"With OSA" vs "Without OSA". The target they actually trained on was
`Sleep_Label`, which the generating code documents as "0=awake, 1=asleep"
(`regenerate_pipeline.py:121`, in git history). It was a sleep/wake classifier
presented as an OSA detector, and its own numbers gave SpO2 -- the defining
marker of apnoea -- an importance of 0.0019.

WHAT THIS SCRIPT DOES
---------------------
The source export is identified: recording start 22:47:44 and 8.9 h duration
match **UCDDB011** (PSG start 22:47:38, 7.5 h, recorded AHI 8) in
`dataset_ucddb_v2`. So the correct labels exist -- the expert respiratory event
list for that subject -- and are attached here in place of the sleep stage.

Because the exported CSVs carry ABSOLUTE wall-clock timestamps, alignment is
exact rather than inferred, and is checked against the recorded AHI.

TWO HONEST CONSTRAINTS
----------------------
1. This is **one subject, one night**. No between-subject generalisation can be
   claimed, and no subject-level OSA classification is possible from n=1.
2. Consecutive 30 s epochs from one night are strongly autocorrelated. A random
   split would leak neighbouring epochs across the fold boundary, so this uses
   **contiguous block CV**: five time-ordered blocks, each held out in turn.

Outputs -> cross_dataset/structured/ucddb011_epochs.{parquet,csv}
           cross_dataset/results/relabelled_*.csv
"""
from pathlib import Path
import numpy as np, pandas as pd, re, sys, warnings
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                             precision_score, f1_score, accuracy_score,
                             balanced_accuracy_score, confusion_matrix)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CD = ROOT / "cross_dataset"
RAW, S, R = CD / "raw", CD / "structured", CD / "results"
for d in (S, R):
    d.mkdir(parents=True, exist_ok=True)
UC = ROOT / "dataset_ucddb_v2" / "raw"
SUBJ = "ucddb011"
EPOCH = 30.0
SEED = 42
EVENT_TYPES = {"APNEA-O", "APNEA-C", "APNEA-M", "HYP-O", "HYP-C", "HYP-M"}

rep = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); rep.append(s)


def secs(hms):
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_signal(fname, col):
    d = pd.read_csv(RAW / fname)
    t = d["absolute position (hh:mm:ss.ms)"].map(secs).to_numpy()
    return t, pd.to_numeric(d[col], errors="coerce").to_numpy(float)


def unwrap(t):
    """Recording crosses midnight; make time monotonic."""
    t = t.copy()
    jump = np.where(np.diff(t) < -12 * 3600)[0]
    for j in jump:
        t[j + 1:] += 24 * 3600
    return t


def main():
    stage = pd.read_csv(RAW / "50_sleep_stage.csv")
    st_t = unwrap(stage["absolute position (hh:mm:ss.ms)"].map(secs).to_numpy())
    stages = stage['Default Staging Set ("stage")'].astype(str).str.strip().to_numpy()
    n_ep = len(stages)
    t0 = st_t[0]
    p(f"source export: {n_ep} epochs from {stage['absolute position (hh:mm:ss.ms)'].iloc[0]} "
      f"({n_ep*EPOCH/3600:.2f} h)")

    # ---- correct labels: real respiratory events for UCDDB011 ----
    ev = []
    for ln in (UC / f"{SUBJ}_respevt.txt").read_text(errors="replace").splitlines()[3:]:
        m = re.match(r"\s*(\d{1,2}:\d{2}:\d{2})\s+(\S+)\s+(.*)$", ln)
        if not m:
            continue
        tm, typ, rest = m.groups()
        if typ not in EVENT_TYPES:
            continue
        d = re.match(r"\s*(?:\S+\s+)??(\d+)\s", rest)
        ev.append((secs(tm), float(d.group(1)) if d else 15.0, typ))
    p(f"UCDDB011 scored respiratory events: {len(ev)}")

    y = np.zeros(n_ep, dtype=int)
    for t_abs, dur, _ in ev:
        rel = t_abs - t0
        if rel < -12 * 3600:
            rel += 24 * 3600
        if rel < -dur:
            continue
        a = max(int(rel // EPOCH), 0); b = min(int(np.ceil((rel + dur) / EPOCH)), n_ep)
        if a < n_ep:
            y[a:max(b, a + 1)] = 1

    # ---- features per epoch ----
    feats = {}
    for fname, col, pre in [("50_HR.csv", 'Heart Rate ("bpm")', "hr"),
                            ("50_SpO2.csv", 'OSat ("%")', "spo2"),
                            ("50_Flow_DR.csv", "Flow_DR", "flow")]:
        t, x = load_signal(fname, col)
        t = unwrap(t)
        idx = np.floor((t - t0) / EPOCH).astype(int)
        ok = (idx >= 0) & (idx < n_ep) & np.isfinite(x)
        g = pd.Series(x[ok]).groupby(idx[ok])
        for stat, fn in [("mean", g.mean), ("std", g.std), ("min", g.min),
                         ("max", g.max)]:
            feats[f"{pre}_{stat}"] = fn().reindex(range(n_ep)).to_numpy()
        feats[f"{pre}_range"] = feats[f"{pre}_max"] - feats[f"{pre}_min"]

    df = pd.DataFrame(feats)
    b = pd.Series(df.spo2_mean).rolling(4, min_periods=1).max()
    df["spo2_desat"] = b - df.spo2_min
    df["spo2_below90"] = (df.spo2_min < 90).astype(float)
    bf = pd.Series(df.flow_std).rolling(4, min_periods=1).max()
    df["flow_rel_amp"] = df.flow_std / bf.replace(0, np.nan)
    for L in (1, 2):
        roll = df[["spo2_desat", "spo2_min", "flow_rel_amp", "hr_mean", "hr_std"]] \
            .rolling(2 * L + 1, center=True, min_periods=1).mean()
        df = pd.concat([df, roll.add_suffix(f"_ctx{L}")], axis=1)

    df.insert(0, "epoch", np.arange(n_ep))
    df["stage"] = stages
    df["asleep"] = (~pd.Series(stages).isin(["W"])).astype(int).to_numpy()
    df["event"] = y
    df["sleep_label_LEGACY"] = df["asleep"]     # the old, wrong target, kept for contrast
    df = df.replace([np.inf, -np.inf], np.nan)

    # ---- validation ----
    ok = True
    def chk(name, val, passed):
        nonlocal ok; ok &= bool(passed); p(f"  [{'PASS' if passed else 'FAIL'}] {name}: {val}")

    p("\n-- validation --")
    sleep_h = df.asleep.sum() * EPOCH / 3600
    ai = y.sum() / sleep_h
    chk("event epochs present", int(y.sum()), y.sum() > 0)
    chk("derived event index near recorded AHI 8",
        f"{ai:.1f} events/h vs AHI 8", 3 <= ai <= 20)
    chk("events concentrated in sleep",
        f"{df[df.asleep==1].event.mean():.1%} vs {df[df.asleep==0].event.mean():.1%}",
        df[df.asleep == 1].event.mean() > df[df.asleep == 0].event.mean())
    chk("new target differs from legacy target",
        f"agreement {(df.event==df.sleep_label_LEGACY).mean():.1%}",
        (df.event == df.sleep_label_LEGACY).mean() < 0.9)
    p(f"\n  epochs {n_ep} | event rate {y.mean():.1%} | legacy 'OSA' target rate "
      f"{df.sleep_label_LEGACY.mean():.1%}")

    FE = [c for c in df.columns
          if c not in ("epoch", "event", "stage", "asleep", "sleep_label_LEGACY")
          and df[c].dtype.kind in "fi"]
    X = df[FE].to_numpy(float)

    # ---- contiguous block CV ----
    p("\n-- models (5 contiguous time blocks; a random split would leak neighbours) --")
    blocks = np.array_split(np.arange(n_ep), 5)
    models = {
        "LogisticRegression_L2": Pipeline([
            ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced",
                                     random_state=SEED))]),
        "LogisticRegression_L1": Pipeline([
            ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=5000, C=0.5, penalty="l1", solver="liblinear",
                                     class_weight="balanced", random_state=SEED))]),
        "RandomForest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                         class_weight="balanced_subsample",
                                         random_state=SEED, n_jobs=-1))]),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([("m", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
            scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)),
            eval_metric="aucpr", random_state=SEED, n_jobs=-1))])
    except ImportError:
        pass

    rows, oof = [], {}
    for name, mod in models.items():
        o = np.zeros(n_ep)
        for bi in blocks:
            tr = np.setdiff1d(np.arange(n_ep), bi)
            o[bi] = mod.fit(X[tr], y[tr]).predict_proba(X[bi])[:, 1]
        oof[name] = o
        pred = (o >= .5).astype(int)
        rows.append(dict(model=name, target="respiratory_event",
                         recall=recall_score(y, pred, zero_division=0),
                         precision=precision_score(y, pred, zero_division=0),
                         f1=f1_score(y, pred, zero_division=0),
                         pr_auc=average_precision_score(y, o), roc_auc=roc_auc_score(y, o),
                         accuracy=accuracy_score(y, pred),
                         balanced_accuracy=balanced_accuracy_score(y, pred)))
        p(f"  {name:24s} recall={rows[-1]['recall']:.3f} PR-AUC={rows[-1]['pr_auc']:.4f} "
          f"ROC-AUC={rows[-1]['roc_auc']:.4f}")
    rows.append(dict(model="_baseline_majority", target="respiratory_event", recall=0.0,
                     precision=0.0, f1=0.0, pr_auc=y.mean(), roc_auc=0.5,
                     accuracy=max(y.mean(), 1 - y.mean()), balanced_accuracy=0.5))
    md = pd.DataFrame(rows); md.to_csv(R / "relabelled_metrics.csv", index=False)
    best = md[md.model != "_baseline_majority"].set_index("model").pr_auc.idxmax()
    p(f"\n  best: {best} (baseline PR-AUC={y.mean():.4f})")
    pd.DataFrame(confusion_matrix(y, (oof[best] >= .5).astype(int)),
                 index=["true_normal", "true_event"],
                 columns=["pred_normal", "pred_event"]).to_csv(R / "relabelled_confusion.csv")

    # ---- the contrast that makes the mislabelling concrete ----
    p("\n-- same features, old target vs corrected target --")
    cmp_rows = []
    m = models["LogisticRegression_L2"]
    for tgt, yy in [("sleep_vs_wake (LEGACY target)", df.sleep_label_LEGACY.to_numpy()),
                    ("respiratory_event (CORRECT)", y)]:
        o = np.zeros(n_ep)
        for bi in blocks:
            tr = np.setdiff1d(np.arange(n_ep), bi)
            o[bi] = m.fit(X[tr], yy[tr]).predict_proba(X[bi])[:, 1]
        cmp_rows.append(dict(target=tgt, base_rate=yy.mean(),
                             accuracy=accuracy_score(yy, (o >= .5).astype(int)),
                             balanced_accuracy=balanced_accuracy_score(yy, (o >= .5).astype(int)),
                             roc_auc=roc_auc_score(yy, o),
                             pr_auc=average_precision_score(yy, o)))
        p(f"  {tgt:32s} base={yy.mean():.3f} acc={cmp_rows[-1]['accuracy']:.3f} "
          f"bal_acc={cmp_rows[-1]['balanced_accuracy']:.3f} ROC-AUC={cmp_rows[-1]['roc_auc']:.4f}")
    pd.DataFrame(cmp_rows).to_csv(R / "relabelled_target_comparison.csv", index=False)

    # feature importance on the corrected target
    rf = models["RandomForest"].fit(X, y)
    fi = pd.DataFrame(dict(feature=FE, rf_impurity=rf.named_steps["m"].feature_importances_))
    fi["univariate_auc"] = [max(roc_auc_score(y, np.nan_to_num(X[:, i], nan=np.nanmedian(X[:, i]))),
                                1 - roc_auc_score(y, np.nan_to_num(X[:, i], nan=np.nanmedian(X[:, i]))))
                            for i in range(len(FE))]
    fi = fi.sort_values("rf_impurity", ascending=False)
    fi.to_csv(R / "relabelled_feature_importance.csv", index=False)
    p("\ntop features on the corrected target:")
    p(fi.head(8).round(4).to_string(index=False))
    p(f"\n  SpO2 share of total importance: "
      f"{fi[fi.feature.str.startswith('spo2')].rf_impurity.sum():.1%}  "
      f"(legacy analysis gave SpO2 importance 0.0019)")

    df.to_parquet(S / "ucddb011_epochs.parquet", index=False)
    df.to_csv(S / "ucddb011_epochs.csv", index=False)
    (R / "relabelled_report.txt").write_text("\n".join(rep), encoding="utf-8")
    p(f"\n[write] {S/'ucddb011_epochs.parquet'}, {R/'relabelled_metrics.csv'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
