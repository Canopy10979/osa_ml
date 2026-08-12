"""Stages 3-4 for dataset_apnea_hrv: train three model families and evaluate.

Models (identical splits, seed recorded):
  * Logistic Regression, L1 and L2 -- the interpretable baseline
  * Random Forest
  * XGBoost (falls back to sklearn HistGradientBoosting if unavailable)

Splits: StratifiedGroupKFold(5) grouped by SUBJECT. Subjects contribute ~400
minutes each, so a random minute split would leak badly. Additionally the
database's own learning/test partition (L/T) is scored as a true holdout --
those test subjects are never seen in any fold.

Leads with recall and PR-AUC: a missed apnoea is the costly error.

Outputs -> results/: metrics.csv, metrics.json, confusion_*.csv,
classification_report_*.txt, feature_importance.csv, calibration.csv,
per_subject_metrics.csv, oof_predictions.csv, holdout_metrics.csv
Outputs -> models/: *.joblib, params.json
"""
from pathlib import Path
import numpy as np, pandas as pd, json, time, warnings, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report, cohen_kappa_score,
                             brier_score_loss)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset_apnea_hrv"
S, R, M = DS / "structured", DS / "results", DS / "models"
for d in (R, M):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
rep = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); rep.append(s)

df = pd.read_parquet(S / "minute_features.parquet")
META = {"subject", "minute", "apnea", "osa", "group", "split"}
FEATS = [c for c in df.columns if c not in META and df[c].dtype.kind in "fi"]
X = df[FEATS].to_numpy(float)
y = df["apnea"].to_numpy()
g = df["subject"].to_numpy()

p("=" * 78); p("STAGE 3-4 — MODELS (dataset_apnea_hrv)"); p("=" * 78)
p(f"{len(df):,} minutes | {df.subject.nunique()} subjects | {len(FEATS)} features")
p(f"class balance: {y.mean():.3%} apnoea  (majority baseline accuracy {max(y.mean(),1-y.mean()):.4f})")
p(f"CV: StratifiedGroupKFold(5) grouped by subject — every subject held out exactly once.")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False
    p("xgboost unavailable — falling back to HistGradientBoosting")

pos_w = float((y == 0).sum() / max((y == 1).sum(), 1))

def build():
    m = {
        "LogisticRegression_L2": Pipeline([
            ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=5000, C=0.5, penalty="l2",
                                     class_weight="balanced", random_state=SEED))]),
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
    if HAVE_XGB:
        m["XGBoost"] = Pipeline([
            ("m", XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                                subsample=0.9, colsample_bytree=0.8,
                                scale_pos_weight=pos_w, eval_metric="aucpr",
                                random_state=SEED, n_jobs=-1))])
    else:
        m["HistGradientBoosting"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                 class_weight="balanced",
                                                 random_state=SEED))])
    return m


def metrics(name, yt, prob, secs=np.nan):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return dict(model=name,
                accuracy=accuracy_score(yt, pred),
                balanced_accuracy=balanced_accuracy_score(yt, pred),
                precision=precision_score(yt, pred, zero_division=0),
                recall=recall_score(yt, pred, zero_division=0),
                specificity=tn / (tn + fp) if tn + fp else np.nan,
                f1=f1_score(yt, pred, zero_division=0),
                roc_auc=roc_auc_score(yt, prob),
                pr_auc=average_precision_score(yt, prob),
                kappa=cohen_kappa_score(yt, pred),
                brier=brier_score_loss(yt, prob),
                seconds=round(secs, 1))


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X, y, g))
oof = {}
rows = []

p("\n-- out-of-fold (grouped by subject) --")
for name, mod in build().items():
    t0 = time.time()
    o = np.zeros(len(y))
    for tr, te in folds:
        mod.fit(X[tr], y[tr])
        o[te] = mod.predict_proba(X[te])[:, 1]
    secs = time.time() - t0
    oof[name] = o
    rows.append(metrics(name, y, o, secs))
    p(f"  {name:24s} recall={rows[-1]['recall']:.3f}  PR-AUC={rows[-1]['pr_auc']:.4f}  "
      f"ROC-AUC={rows[-1]['roc_auc']:.4f}  ({secs:.0f}s)")

    cm = confusion_matrix(y, (o >= 0.5).astype(int), labels=[0, 1])
    pd.DataFrame(cm, index=["true_normal", "true_apnoea"],
                 columns=["pred_normal", "pred_apnoea"]).to_csv(R / f"confusion_{name}.csv")
    (R / f"classification_report_{name}.txt").write_text(
        classification_report(y, (o >= 0.5).astype(int),
                              target_names=["normal", "apnoea"], digits=4), encoding="utf-8")

# majority baseline
base = np.full(len(y), y.mean())
b = metrics("_baseline_majority", y, base * 0)
b["pr_auc"] = y.mean(); b["roc_auc"] = 0.5
rows.append(b)

md = pd.DataFrame(rows)
md.to_csv(R / "metrics.csv", index=False)
(R / "metrics.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")

best = md[md.model != "_baseline_majority"].set_index("model")["pr_auc"].idxmax()
p(f"\n  best by PR-AUC: {best}")
p(f"  majority baseline: accuracy={b['accuracy']:.4f} PR-AUC={b['pr_auc']:.4f}")

# ---- official L/T holdout: test subjects never seen in training ----
p("\n-- official L/T holdout (train on L subjects, test on T subjects) --")
trm, tem = (df.split == "L").to_numpy(), (df.split == "T").to_numpy()
hold = []
for name, mod in build().items():
    mod.fit(X[trm], y[trm])
    pr = mod.predict_proba(X[tem])[:, 1]
    h = metrics(name, y[tem], pr)
    hold.append({k: h[k] for k in ("model", "recall", "pr_auc", "roc_auc", "f1", "accuracy", "balanced_accuracy")})
    p(f"  {name:24s} recall={h['recall']:.3f}  PR-AUC={h['pr_auc']:.4f}  ROC-AUC={h['roc_auc']:.4f}")
    joblib.dump(mod, M / f"{name}.joblib")
pd.DataFrame(hold).to_csv(R / "holdout_metrics.csv", index=False)

# ---- calibration (best model, OOF) ----
ob = oof[best]
q = pd.qcut(ob, 10, labels=False, duplicates="drop")
pd.DataFrame({"mean_predicted": pd.Series(ob).groupby(q).mean(),
              "observed_fraction": pd.Series(y).groupby(q).mean()}).to_csv(
    R / "calibration.csv", index=False)

# ---- per-subject behaviour ----
sm = []
for s in df.subject.unique():
    m = df.subject.values == s
    if y[m].sum() in (0, m.sum()):
        auc = np.nan
    else:
        auc = roc_auc_score(y[m], ob[m])
    sm.append(dict(subject=s, group=df.group[m].iloc[0], n_min=int(m.sum()),
                   apnea_rate=float(y[m].mean()),
                   recall=recall_score(y[m], (ob[m] >= .5).astype(int), zero_division=0),
                   accuracy=accuracy_score(y[m], (ob[m] >= .5).astype(int)), roc_auc=auc))
pd.DataFrame(sm).to_csv(R / "per_subject_metrics.csv", index=False)

pd.DataFrame({"subject": g, "minute": df.minute, "y": y,
              **{f"p_{k}": v for k, v in oof.items()}}).to_csv(
    R / "oof_predictions.csv", index=False)

# ---- importances ----
p("\n-- feature importance --")
lr = build()["LogisticRegression_L2"].fit(X, y)
coef = lr.named_steps["m"].coef_[0]
rf = build()["RandomForest"].fit(X, y)
imp = rf.named_steps["m"].feature_importances_
perm = permutation_importance(rf, X[::10], y[::10], n_repeats=3,
                              random_state=SEED, n_jobs=-1, scoring="average_precision")
fi = pd.DataFrame(dict(feature=FEATS, lr_coef=coef, lr_odds_ratio=np.exp(coef),
                       rf_impurity=imp, rf_permutation=perm.importances_mean))
if HAVE_XGB:
    xg = build()["XGBoost"].fit(X, y)
    fi["xgb_gain"] = xg.named_steps["m"].feature_importances_
    fi = fi.sort_values("xgb_gain", ascending=False)
else:
    fi = fi.sort_values("rf_impurity", ascending=False)
fi.to_csv(R / "feature_importance.csv", index=False)
p(fi.head(12).round(4).to_string(index=False))
p("\nSHAP not installed — XGB gain + RF permutation importance reported instead.")

json.dump({"seed": SEED, "cv": "StratifiedGroupKFold(5) by subject",
           "holdout": "official L/T subject partition",
           "n_features": len(FEATS), "features": FEATS,
           "xgboost": HAVE_XGB, "scale_pos_weight": pos_w},
          open(M / "params.json", "w"), indent=2)

(R / "stage3_4_report.txt").write_text("\n".join(rep), encoding="utf-8")
p(f"\nwrote results to {R}")
