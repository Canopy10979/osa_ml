"""Stages 3-6 for dataset_ucddb_v2: train, evaluate, infer, and self-validate.

Three model families on identical subject-grouped splits: Logistic Regression
(L1 and L2), Random Forest, XGBoost. Leads with recall and PR-AUC.

Two questions are asked:
  * per-epoch  -- is a respiratory event happening in this 30 s window?
  * per-subject -- does this person have OSA (AHI >= 15)?

The second is reported with cross-validated metrics only and NO held-out test
set: 25 subjects with 11 in the minority class is below the threshold at which
a test split carries information (see osa-ml-skill.md).

Outputs -> results/ and models/.
"""
from pathlib import Path
import numpy as np, pandas as pd, json, time, warnings, joblib
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneOut
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report, cohen_kappa_score,
                             brier_score_loss, mean_absolute_error)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset_ucddb_v2"
S, R, M = DS / "structured", DS / "results", DS / "models"
for d in (R, M):
    d.mkdir(parents=True, exist_ok=True)
SEED = 42

rep = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); rep.append(s)

df = pd.read_parquet(S / "epoch_features.parquet")
sub = pd.read_csv(S / "subject_level.csv")

# `ahi` and `osa_15` are subject-level outcomes and `stage` is expert scoring
# from the same read -- all excluded. `asleep` is derived from `stage` and is
# kept: knowing the patient is asleep is available at inference time from the
# same PSG and is not the outcome.
DROP = {"subject", "epoch", "event", "ahi", "osa_15", "stage"}
FEATS = [c for c in df.columns if c not in DROP and df[c].dtype.kind in "fi"]
X = df[FEATS].to_numpy(float); y = df["event"].to_numpy(); g = df["subject"].to_numpy()

p("=" * 78); p("STAGE 3-4 — MODELS (dataset_ucddb_v2)"); p("=" * 78)
p(f"{len(df):,} epochs | {df.subject.nunique()} subjects | {len(FEATS)} features")
p(f"class balance: {y.mean():.3%} event  (majority baseline accuracy {max(y.mean(),1-y.mean()):.4f})")
p("CV: StratifiedGroupKFold(5) grouped by subject — every subject held out exactly once.")

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_XGB = False

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
        m["XGBoost"] = Pipeline([("m", XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="aucpr",
            random_state=SEED, n_jobs=-1))])
    else:
        m["HistGradientBoosting"] = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                 class_weight="balanced", random_state=SEED))])
    return m


def metrics(name, yt, prob, secs=np.nan):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return dict(model=name, accuracy=accuracy_score(yt, pred),
                balanced_accuracy=balanced_accuracy_score(yt, pred),
                precision=precision_score(yt, pred, zero_division=0),
                recall=recall_score(yt, pred, zero_division=0),
                specificity=tn / (tn + fp) if tn + fp else np.nan,
                f1=f1_score(yt, pred, zero_division=0),
                roc_auc=roc_auc_score(yt, prob),
                pr_auc=average_precision_score(yt, prob),
                kappa=cohen_kappa_score(yt, pred),
                brier=brier_score_loss(yt, prob), seconds=round(secs, 1))


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X, y, g))
oof, rows = {}, []
p("\n-- out-of-fold (grouped by subject) --")
for name, mod in build().items():
    t0 = time.time(); o = np.zeros(len(y))
    for tr, te in folds:
        mod.fit(X[tr], y[tr]); o[te] = mod.predict_proba(X[te])[:, 1]
    secs = time.time() - t0
    oof[name] = o; rows.append(metrics(name, y, o, secs))
    p(f"  {name:24s} recall={rows[-1]['recall']:.3f}  PR-AUC={rows[-1]['pr_auc']:.4f}  "
      f"ROC-AUC={rows[-1]['roc_auc']:.4f}  ({secs:.0f}s)")
    pd.DataFrame(confusion_matrix(y, (o >= .5).astype(int), labels=[0, 1]),
                 index=["true_normal", "true_event"],
                 columns=["pred_normal", "pred_event"]).to_csv(R / f"confusion_{name}.csv")
    (R / f"classification_report_{name}.txt").write_text(
        classification_report(y, (o >= .5).astype(int),
                              target_names=["normal", "event"], digits=4), encoding="utf-8")
    mod.fit(X, y); joblib.dump(mod, M / f"{name}.joblib")

b = metrics("_baseline_majority", y, np.zeros(len(y)))
b["pr_auc"] = y.mean(); b["roc_auc"] = 0.5
rows.append(b)
md = pd.DataFrame(rows); md.to_csv(R / "metrics.csv", index=False)
(R / "metrics.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
BEST = md[md.model != "_baseline_majority"].set_index("model")["pr_auc"].astype(float).idxmax()
p(f"\n  best by PR-AUC: {BEST}   (baseline PR-AUC={b['pr_auc']:.4f}, acc={b['accuracy']:.4f})")

ob = oof[BEST]
q = pd.qcut(ob, 10, labels=False, duplicates="drop")
pd.DataFrame({"mean_predicted": pd.Series(ob).groupby(q).mean(),
              "observed_fraction": pd.Series(y).groupby(q).mean()}).to_csv(
    R / "calibration.csv", index=False)
pd.DataFrame({"subject": g, "epoch": df.epoch, "y": y,
              **{f"p_{k}": v for k, v in oof.items()}}).to_csv(
    R / "oof_predictions.csv", index=False)

# ---- importances ----
p("\n-- feature importance --")
lr = build()["LogisticRegression_L2"].fit(X, y)
coef = lr.named_steps["m"].coef_[0]
rf = build()["RandomForest"].fit(X, y)
perm = permutation_importance(rf, X[::8], y[::8], n_repeats=3, random_state=SEED,
                              n_jobs=-1, scoring="average_precision")
fi = pd.DataFrame(dict(feature=FEATS, lr_coef=coef, lr_odds_ratio=np.exp(coef),
                       rf_impurity=rf.named_steps["m"].feature_importances_,
                       rf_permutation=perm.importances_mean))
if HAVE_XGB:
    fi["xgb_gain"] = build()["XGBoost"].fit(X, y).named_steps["m"].feature_importances_
    fi = fi.sort_values("xgb_gain", ascending=False)
else:
    fi = fi.sort_values("rf_impurity", ascending=False)
fi.to_csv(R / "feature_importance.csv", index=False)
p(fi.head(10).round(4).to_string(index=False))

# ---- Stage 5: subject level ----
p("\n" + "=" * 78); p("STAGE 5 — SUBJECT-LEVEL OSA (AHI >= 15)"); p("=" * 78)
sp = (pd.DataFrame({"subject": g, "pred": (ob >= .5).astype(int), "y": y})
      .groupby("subject").agg(n_ep=("y", "size"), pred_ep=("pred", "sum"),
                              true_ep=("y", "sum")).reset_index())
sp = sp.merge(sub[["subject", "sleep_hours", "ahi", "osa_15", "epworth", "BMI", "Age"]],
              on="subject")
sp["ai_pred"] = sp.pred_ep / sp.sleep_hours
r_p = stats.pearsonr(sp.ahi, sp.ai_pred)
p(f"predicted event index vs recorded AHI: Pearson r={r_p[0]:.3f} (p={r_p[1]:.2g})  "
  f"MAE={mean_absolute_error(sp.ahi, sp.ai_pred):.2f}")
auc_sub = roc_auc_score(sp.osa_15, sp.ai_pred)
p(f"as a ranking score for OSA (AHI>=15): ROC-AUC={auc_sub:.4f}")

cands = np.arange(2, 60, 0.5)
bal = [balanced_accuracy_score(sp.osa_15, (sp.ai_pred >= t).astype(int)) for t in cands]
thr = float(cands[int(np.argmax(bal))])
sp["osa_pred"] = (sp.ai_pred >= thr).astype(int)
p(f"at best threshold {thr:g}: balanced_acc={balanced_accuracy_score(sp.osa_15, sp.osa_pred):.3f} "
  f"sens={recall_score(sp.osa_15, sp.osa_pred):.3f} "
  f"spec={recall_score(1-sp.osa_15, 1-sp.osa_pred):.3f}")
p(f"  (threshold fitted on these same {len(sp)} subjects — optimistic; the "
  f"threshold-free ROC-AUC {auc_sub:.3f} is the honest figure)")
p(f"confusion:\n{confusion_matrix(sp.osa_15, sp.osa_pred)}")
sp.to_csv(R / "subject_level_predictions.csv", index=False)

# leave-one-subject-out on the pooled index: no test set is manufactured, the
# threshold is refit inside each fold so the held-out subject never informs it
loo_pred = []
for i in range(len(sp)):
    tr = sp.drop(index=i)
    b2 = [balanced_accuracy_score(tr.osa_15, (tr.ai_pred >= t).astype(int)) for t in cands]
    loo_pred.append(int(sp.ai_pred.iloc[i] >= cands[int(np.argmax(b2))]))
p(f"\nleave-one-subject-out (threshold refit each fold): "
  f"balanced_acc={balanced_accuracy_score(sp.osa_15, loo_pred):.3f} "
  f"acc={accuracy_score(sp.osa_15, loo_pred):.3f}")

# ---- Stage 6 ----
p("\n" + "=" * 78); p("STAGE 6 — VALIDATING THE ANALYSIS"); p("=" * 78)
def run(yy, seed):
    c = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    Xf = np.nan_to_num(X, nan=np.nanmedian(X)); o = np.zeros(len(yy))
    for tr, te in c.split(Xf, yy, g):
        if HAVE_XGB:
            mm = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                               subsample=0.9, colsample_bytree=0.8, scale_pos_weight=pos_w,
                               eval_metric="aucpr", random_state=seed, n_jobs=-1)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier as H
            mm = H(max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=seed)
        o[te] = mm.fit(Xf[tr], yy[tr]).predict_proba(Xf[te])[:, 1]
    return o

p("\n6a. LABEL-SHUFFLE TEST")
rng = np.random.default_rng(SEED); sh = []
for tag, ysh in [("global shuffle", rng.permutation(y)),
                 ("within-subject shuffle",
                  pd.Series(y).groupby(g).transform(
                      lambda s: rng.permutation(s.to_numpy())).to_numpy())]:
    o = run(ysh, SEED)
    sh.append(dict(test=tag, pr_auc=average_precision_score(ysh, o),
                   roc_auc=roc_auc_score(ysh, o), base_rate=ysh.mean()))
    p(f"  {tag:24s} PR-AUC={sh[-1]['pr_auc']:.4f} (chance={ysh.mean():.4f})  "
      f"ROC-AUC={sh[-1]['roc_auc']:.4f}")
pd.DataFrame(sh).to_csv(R / "shuffle_test.csv", index=False)
REAL_ROC = float(md.loc[md.model == BEST, "roc_auc"].iloc[0])
p(f"  real labels              ROC-AUC={REAL_ROC:.4f}")
ns = {r["test"]: r for r in sh}
leak = abs(ns["global shuffle"]["roc_auc"] - 0.5) < 0.05
p(f"  VERDICT (global shuffle): {'PASS — no leakage.' if leak else 'FAIL — investigate.'}")
wr = ns["within-subject shuffle"]["roc_auc"]
p(f"  DIAGNOSTIC (within-subject shuffle) ROC-AUC={wr:.4f} -> "
  f"~{(wr-0.5)/(REAL_ROC-0.5):.0%} of skill is between-subject")

p("\n6b. SEED VARIANCE")
sv = []
for sd in (42, 7, 2024):
    o = run(y, sd)
    sv.append(dict(seed=sd, pr_auc=average_precision_score(y, o),
                   roc_auc=roc_auc_score(y, o), recall=recall_score(y, (o >= .5).astype(int))))
    p(f"  seed={sd:5d} PR-AUC={sv[-1]['pr_auc']:.4f} ROC-AUC={sv[-1]['roc_auc']:.4f}")
sv = pd.DataFrame(sv); sv.to_csv(R / "seed_variance.csv", index=False)
p(f"  PR-AUC mean={sv.pr_auc.mean():.4f} sd={sv.pr_auc.std():.4f}")

json.dump({"seed": SEED, "cv": "StratifiedGroupKFold(5) by subject",
           "n_features": len(FEATS), "features": FEATS, "xgboost": HAVE_XGB,
           "subject_threshold": thr,
           "note": "no held-out test set at subject level: 25 subjects / 11 minority"},
          open(M / "params.json", "w"), indent=2)
(R / "stage3_6_report.txt").write_text("\n".join(rep), encoding="utf-8")
p(f"\nwrote results to {R}")
