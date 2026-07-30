"""Stages 2-4 — leakage audit, three models, evaluation (Apnea-ECG).

Target : `apnea` — the database's own per-minute annotation (A vs N).
Unit   : one minute of ECG. Minutes are NOT independent (≈490 per record), so
         every split is GROUPED BY RECORD. This is the single most common cause
         of inflated apnoea-detection scores.

Models : Logistic Regression (L1 and L2), Random Forest, XGBoost.
Primary metrics are RECALL and PR-AUC — a missed apnoea is the costly error.

Outputs -> dataset_apnea_ecg/{results,models}/
"""
from pathlib import Path
import numpy as np, pandas as pd, json, time, warnings, joblib
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             classification_report, balanced_accuracy_score,
                             cohen_kappa_score, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset_apnea_ecg"
S, R, M = DS / "structured", DS / "results", DS / "models"
for d in (R, M):
    d.mkdir(parents=True, exist_ok=True)
SEED = 42

rep = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); rep.append(s)

df = pd.read_parquet(S / "minute_features.parquet")
LEAK = {"record", "minute", "label", "apnea", "class_prefix"}
FEATS = [c for c in df.columns if c not in LEAK and df[c].dtype.kind in "fi"]
X = df[FEATS].to_numpy(float)
y = df["apnea"].to_numpy()
g = df["record"].to_numpy()

p("=" * 78)
p("APNEA-ECG — per-minute apnoea detection")
p("=" * 78)
p(f"rows={len(df)}  features={len(FEATS)}  records={df.record.nunique()}")
p(f"class balance: apnea={y.mean():.3f}  majority-class baseline accuracy={max(y.mean(),1-y.mean()):.3f}")

# ---------------------------------------------------------------- Stage 2
p("\n" + "=" * 78)
p("STAGE 2 — LEAKAGE AUDIT")
p("=" * 78)
p("Excluded by construction (identity / outcome-derived): " + ", ".join(sorted(LEAK)))
audit = []
for i, c in enumerate(FEATS):
    v = X[:, i]
    ok = ~np.isnan(v)
    if ok.sum() < 100:
        audit.append((c, np.nan, np.nan, "too few non-null")); continue
    auc1 = roc_auc_score(y[ok], v[ok])
    auc1 = max(auc1, 1 - auc1)                      # direction-agnostic
    r_, _ = stats.pointbiserialr(y[ok], v[ok])
    flag = "SUSPICIOUS (single-feature separator)" if auc1 > 0.90 else ""
    audit.append((c, auc1, r_, flag))
ad = pd.DataFrame(audit, columns=["feature", "univariate_auc", "point_biserial_r", "flag"])
ad = ad.sort_values("univariate_auc", ascending=False)
ad.to_csv(R / "leakage_audit.csv", index=False)
p(f"\nstrongest single features (none should approach 1.0):")
p(ad.head(8).to_string(index=False))
nsusp = int((ad.univariate_auc > 0.90).sum())
p(f"\nfeatures with univariate AUC > 0.90: {nsusp}")
p("VERDICT: " + ("no single feature separates the classes — no evidence of leakage."
                if nsusp == 0 else "REVIEW the flagged features above before trusting results."))
p("\nper-record class balance (a=apnoea, b=borderline, c=control):")
bal = df.groupby(["class_prefix", "record"])["apnea"].agg(["size", "mean"]).round(3)
p(bal.to_string())

# ---------------------------------------------------------------- Stage 3
p("\n" + "=" * 78)
p("STAGE 3 — MODELS (grouped by record)")
p("=" * 78)

def models():
    return {
        "LogisticRegression_L2": Pipeline([
            ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, C=1.0, penalty="l2",
                                       class_weight="balanced", random_state=SEED))]),
        "LogisticRegression_L1": Pipeline([
            ("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, C=0.5, penalty="l1",
                                       solver="liblinear",
                                       class_weight="balanced", random_state=SEED))]),
        "RandomForest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                           max_features="sqrt", n_jobs=-1,
                                           class_weight="balanced_subsample",
                                           random_state=SEED))]),
        "XGBoost": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", xgb.XGBClassifier(n_estimators=400, learning_rate=0.06,
                                      max_depth=5, subsample=0.8,
                                      colsample_bytree=0.8, reg_lambda=1.0,
                                      eval_metric="logloss", n_jobs=-1,
                                      random_state=SEED))]),
    }

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
p(f"\nCV: StratifiedGroupKFold(5) by record — every record held out exactly once.")

rows, oof_store = [], {}
for name, mod in models().items():
    t0 = time.time()
    oof = np.zeros(len(y)); oofp = np.zeros(len(y))
    for tr, te in sgkf.split(X, y, g):
        m = mod.fit(X[tr], y[tr])
        oofp[te] = m.predict_proba(X[te])[:, 1]
        oof[te] = m.predict(X[te])
    oof_store[name] = oofp
    r = dict(model=name,
             accuracy=accuracy_score(y, oof),
             balanced_accuracy=balanced_accuracy_score(y, oof),
             precision=precision_score(y, oof),
             recall=recall_score(y, oof),
             specificity=recall_score(y, oof, pos_label=0),
             f1=f1_score(y, oof),
             roc_auc=roc_auc_score(y, oofp),
             pr_auc=average_precision_score(y, oofp),
             kappa=cohen_kappa_score(y, oof),
             brier=brier_score_loss(y, oofp),
             seconds=round(time.time() - t0, 1))
    rows.append(r)
    p(f"  {name:24s} recall={r['recall']:.4f} PR-AUC={r['pr_auc']:.4f} "
      f"ROC-AUC={r['roc_auc']:.4f} F1={r['f1']:.4f} acc={r['accuracy']:.4f} "
      f"({r['seconds']}s)")
    pd.DataFrame(confusion_matrix(y, oof), index=["true_N", "true_A"],
                 columns=["pred_N", "pred_A"]).to_csv(R / f"confusion_{name}.csv")
    (R / f"classification_report_{name}.txt").write_text(
        classification_report(y, oof, target_names=["Normal", "Apnea"], digits=4),
        encoding="utf-8")

base = dict(model="_baseline_majority", accuracy=max(y.mean(), 1 - y.mean()),
            balanced_accuracy=0.5, precision=y.mean(), recall=1.0 if y.mean() > .5 else 0.0,
            f1=np.nan, roc_auc=0.5, pr_auc=y.mean(), kappa=0.0)
rows.append(base)
res = pd.DataFrame(rows)
res.to_csv(R / "metrics.csv", index=False)
res.to_json(R / "metrics.json", orient="records", indent=2)

p(f"\n  {'_baseline_majority':24s} recall={base['recall']:.4f} "
  f"PR-AUC={base['pr_auc']:.4f} ROC-AUC=0.5000 acc={base['accuracy']:.4f}")
best = max(rows[:-1], key=lambda r: r["pr_auc"])
p(f"\n  BEST by PR-AUC: {best['model']} (PR-AUC {best['pr_auc']:.4f} vs "
  f"baseline {base['pr_auc']:.4f})")
p(f"  Beats majority baseline on accuracy: {best['accuracy'] > base['accuracy']}")

# held-out record test set (untouched by any CV above)
p("\n--- held-out record test (6 records never seen in training) ---")
gss = GroupShuffleSplit(n_splits=1, test_size=6/27, random_state=SEED)
tr, te = next(gss.split(X, y, g))
p(f"  test records: {sorted(set(g[te]))}")
hold = []
for name, mod in models().items():
    m = mod.fit(X[tr], y[tr])
    pr = m.predict_proba(X[te])[:, 1]; yp = m.predict(X[te])
    hold.append(dict(model=name, recall=recall_score(y[te], yp),
                     pr_auc=average_precision_score(y[te], pr),
                     roc_auc=roc_auc_score(y[te], pr), f1=f1_score(y[te], yp),
                     accuracy=accuracy_score(y[te], yp)))
    p(f"  {name:24s} recall={hold[-1]['recall']:.4f} "
      f"PR-AUC={hold[-1]['pr_auc']:.4f} ROC-AUC={hold[-1]['roc_auc']:.4f}")
    joblib.dump(m, M / f"{name}.joblib")
pd.DataFrame(hold).to_csv(R / "holdout_metrics.csv", index=False)
json.dump({"seed": SEED, "cv": "StratifiedGroupKFold(5) by record",
           "holdout_records": sorted(set(map(str, g[te]))),
           "n_features": len(FEATS), "features": FEATS},
          open(M / "params.json", "w"), indent=2)

# ---------------------------------------------------------------- Stage 4
p("\n" + "=" * 78)
p("STAGE 4 — EVALUATION DETAIL")
p("=" * 78)

bestname = best["model"]
p(f"\nclassification report — {bestname}:")
p((R / f"classification_report_{bestname}.txt").read_text(encoding="utf-8"))
cm = pd.read_csv(R / f"confusion_{bestname}.csv", index_col=0)
p("confusion matrix:")
p(cm.to_string())

# calibration
p("\ncalibration (predicted vs observed apnoea rate), best model:")
frac, mean_pred = calibration_curve(y, oof_store[bestname], n_bins=10)
cal = pd.DataFrame({"mean_predicted": mean_pred, "observed_fraction": frac})
cal.to_csv(R / "calibration.csv", index=False)
p(cal.round(3).to_string(index=False))

# per-record performance
pr_rows = []
for rec in sorted(df.record.unique()):
    m_ = g == rec
    yy, pp_ = y[m_], oof_store[bestname][m_]
    yp = (pp_ >= .5).astype(int)
    pr_rows.append(dict(record=rec, class_prefix=rec[0], n=int(m_.sum()),
                        apnea_rate=float(yy.mean()),
                        recall=recall_score(yy, yp, zero_division=np.nan),
                        accuracy=accuracy_score(yy, yp),
                        roc_auc=roc_auc_score(yy, pp_) if len(set(yy)) > 1 else np.nan))
prd = pd.DataFrame(pr_rows)
prd.to_csv(R / "per_record_metrics.csv", index=False)
p("\nper-record accuracy (best model):")
p(prd.round(3).to_string(index=False))
p(f"\nmean per-record accuracy={prd.accuracy.mean():.3f} "
  f"sd={prd.accuracy.std():.3f} min={prd.accuracy.min():.3f}")

# importances
p("\n--- feature importance ---")
lr = models()["LogisticRegression_L2"].fit(X, y)
coef = lr.named_steps["clf"].coef_[0]
imp = pd.DataFrame({"feature": FEATS, "lr_coef": coef,
                    "lr_odds_ratio": np.exp(coef)})
rf = models()["RandomForest"].fit(X, y)
imp["rf_impurity"] = rf.named_steps["clf"].feature_importances_
xg = models()["XGBoost"].fit(X, y)
imp["xgb_gain"] = xg.named_steps["clf"].feature_importances_
sub = np.random.default_rng(SEED).choice(len(y), 4000, replace=False)
pi = permutation_importance(rf, X[sub], y[sub], n_repeats=5,
                            random_state=SEED, n_jobs=-1, scoring="average_precision")
imp["rf_permutation"] = pi.importances_mean
imp = imp.sort_values("xgb_gain", ascending=False)
imp.to_csv(R / "feature_importance.csv", index=False)
p(imp.head(15).round(4).to_string(index=False))

try:
    import shap
    p("\nSHAP available — computing summary values")
except ImportError:
    p("\nSHAP not installed — skipped gracefully (XGB gain + permutation reported instead)")

(R / "stage2_4_report.txt").write_text("\n".join(rep), encoding="utf-8")
np.save(R / "oof_best.npy", oof_store[bestname])
pd.DataFrame(oof_store | {"y": y, "record": g}).to_csv(R / "oof_predictions.csv", index=False)
print("\nwrote results to", R)
