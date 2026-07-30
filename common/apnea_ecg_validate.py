"""Stages 5-6 — inference, subject-level OSA, and validation of the analysis.

Stage 5: multicollinearity, model agreement, per-record consistency, and the
         clinically useful output — estimating each subject's apnoea index from
         the per-minute predictions and classifying OSA (AI >= 5).
Stage 6: label-shuffle test (must collapse to chance), seed variance,
         and a per-record failure analysis.
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings, json, time
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (average_precision_score, roc_auc_score, recall_score,
                             accuracy_score, balanced_accuracy_score,
                             mean_absolute_error, confusion_matrix)
import xgboost as xgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset_apnea_ecg"
S, R = DS / "structured", DS / "results"
SEED = 42

rep = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); rep.append(s)

df = pd.read_parquet(S / "minute_features.parquet")
oof = pd.read_csv(R / "oof_predictions.csv")
sub = pd.read_csv(S / "subject_level.csv")

# Headline metrics are READ from the Stage-3/4 artifact, never restated by hand,
# so the shuffle comparison cannot drift out of sync with the run it describes.
_m = pd.read_csv(R / "metrics.csv").set_index("model")
_best = _m.drop(index="_baseline_majority", errors="ignore")["pr_auc"].astype(float).idxmax()
REAL_PR = float(_m.loc[_best, "pr_auc"]); REAL_ROC = float(_m.loc[_best, "roc_auc"])

LEAK = {"record", "minute", "label", "apnea", "class_prefix"}
FEATS = [c for c in df.columns if c not in LEAK and df[c].dtype.kind in "fi"]
X = df[FEATS].to_numpy(float); y = df["apnea"].to_numpy(); g = df["record"].to_numpy()

p("=" * 78)
p("STAGE 5 — CORRELATION, INFERENCE, SUBJECT-LEVEL OSA")
p("=" * 78)

# ---- multicollinearity ----
imp15 = pd.read_csv(R / "feature_importance.csv").head(15)["feature"].tolist()
sub_df = df[imp15].apply(pd.to_numeric, errors="coerce")
sub_df = sub_df.fillna(sub_df.median())
vifs = []
for c in imp15:
    others = [x for x in imp15 if x != c]
    r2 = LinearRegression().fit(sub_df[others], sub_df[c]).score(sub_df[others], sub_df[c])
    vifs.append(dict(feature=c, vif=1.0 / max(1 - r2, 1e-9)))
vd = pd.DataFrame(vifs).sort_values("vif", ascending=False)
vd.to_csv(R / "vif_top15.csv", index=False)
p("\nVIF among the 15 most important features (>10 = heavy collinearity):")
p(vd.round(2).to_string(index=False))
p(f"  features with VIF > 10: {int((vd.vif > 10).sum())} / 15")

cm = df[imp15].corr().abs().to_numpy().copy()
np.fill_diagonal(cm, 0)
p(f"  |r| > 0.95 pairs among top 15: {int((cm > 0.95).sum() // 2)}")

# ---- model agreement ----
mcols = [c for c in oof.columns if c not in ("y", "record")]
p("\nModel agreement (Pearson r between predicted probabilities):")
p(oof[mcols].corr().round(3).to_string())
p("\nWhere models disagree most: fraction of minutes where LR and XGB "
  "predictions straddle 0.5")
lr_, xg_ = oof["LogisticRegression_L2"], oof["XGBoost"]
dis = ((lr_ >= .5) != (xg_ >= .5))
p(f"  disagreement rate = {dis.mean():.3f} ({int(dis.sum())} minutes)")
p(f"  accuracy on agreed minutes   = {accuracy_score(oof.y[~dis], (xg_[~dis]>=.5).astype(int)):.3f}")
p(f"  accuracy on disagreed minutes= {accuracy_score(oof.y[dis], (xg_[dis]>=.5).astype(int)):.3f}")

# ---- SUBJECT-LEVEL OSA (the clinical deliverable) ----
p("\n" + "-" * 78)
p("SUBJECT-LEVEL: estimated apnoea index and OSA classification (AI >= 5)")
p("-" * 78)
p("Per-minute out-of-fold predictions are pooled per record into an estimated")
p("apnoea index (predicted apnoea minutes per hour), then thresholded at the")
p("clinical AI >= 5. Every record was held out when its minutes were predicted.")

est = oof.groupby("record").agg(n_min=("y", "size"),
                                pred_apnea_min=("XGBoost", lambda s: (s >= .5).sum()),
                                true_apnea_min=("y", "sum")).reset_index()
est["hours"] = est.n_min / 60.0
est["ai_pred"] = est.pred_apnea_min / est.hours
est["ai_true"] = est.true_apnea_min / est.hours
est = est.merge(sub[["record", "class_prefix", "osa"]], on="record")
est["osa_pred"] = (est.ai_pred >= 5).astype(int)
est.to_csv(R / "subject_level_predictions.csv", index=False)

r_, pv_ = stats.pearsonr(est.ai_true, est.ai_pred)
rs_, ps_ = stats.spearmanr(est.ai_true, est.ai_pred)
mae = mean_absolute_error(est.ai_true, est.ai_pred)
p(f"\nAI estimation: Pearson r={r_:.3f} (p={pv_:.2g})  Spearman r={rs_:.3f}  MAE={mae:.2f} events/h")
acc = accuracy_score(est.osa, est.osa_pred)
bacc = balanced_accuracy_score(est.osa, est.osa_pred)
p(f"OSA classification (AI>=5): accuracy={acc:.3f}  balanced_accuracy={bacc:.3f}  "
  f"({int(est.osa.sum())}/{len(est)} true positives)")
cmat = confusion_matrix(est.osa, est.osa_pred)
p("confusion (rows=true non-OSA/OSA, cols=pred):")
p(pd.DataFrame(cmat, index=["true_nonOSA", "true_OSA"],
               columns=["pred_nonOSA", "pred_OSA"]).to_string())
p("\nper-subject detail:")
p(est[["record", "class_prefix", "ai_true", "ai_pred", "osa", "osa_pred"]]
  .round(1).to_string(index=False))
mis = est[est.osa != est.osa_pred]
p(f"\nmisclassified subjects: {len(mis)} -> {mis.record.tolist()}")

# ---- Stage 6 ----
p("\n" + "=" * 78)
p("STAGE 6 — VALIDATING THE ANALYSIS ITSELF")
p("=" * 78)

def run_xgb(Xa, ya, ga, seed, shuffle_seed=None):
    mod = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("clf", xgb.XGBClassifier(n_estimators=400, learning_rate=0.06,
                                              max_depth=5, subsample=0.8,
                                              colsample_bytree=0.8, reg_lambda=1.0,
                                              eval_metric="logloss", n_jobs=-1,
                                              random_state=seed))])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                              random_state=shuffle_seed if shuffle_seed is not None else seed)
    o = np.zeros(len(ya))
    for tr, te in cv.split(Xa, ya, ga):
        o[te] = mod.fit(Xa[tr], ya[tr]).predict_proba(Xa[te])[:, 1]
    return o

# 6a. label-shuffle test
p("\n6a. LABEL-SHUFFLE TEST — performance must collapse to chance.")
rng = np.random.default_rng(SEED)
sh_rows = []
for tag, ysh in [("global shuffle", rng.permutation(y)),
                 ("within-record shuffle",
                  pd.Series(y).groupby(g).transform(
                      lambda s: rng.permutation(s.to_numpy())).to_numpy())]:
    o = run_xgb(X, ysh, g, SEED)
    pra = average_precision_score(ysh, o); roc = roc_auc_score(ysh, o)
    sh_rows.append(dict(test=tag, pr_auc=pra, roc_auc=roc, base_rate=ysh.mean()))
    p(f"  {tag:22s} PR-AUC={pra:.4f} (chance={ysh.mean():.4f})  ROC-AUC={roc:.4f} (chance=0.5)")
ns = {r["test"]: r for r in sh_rows}
p(f"  real labels            PR-AUC={REAL_PR:.4f}  ROC-AUC={REAL_ROC:.4f}")

# The leakage verdict rests on the GLOBAL shuffle alone. Within-record shuffling
# permutes labels inside each record, which preserves that record's apnoea rate;
# a grouped model that has learned "this subject breathes like a severe apnoeic"
# therefore still scores above chance without any outcome information having
# leaked into the features. Judging leakage on it conflates between-record signal
# with contamination, so it is reported as a diagnostic, not a pass/fail gate.
leak = abs(ns["global shuffle"]["roc_auc"] - 0.5) < 0.05
p(f"  VERDICT (global shuffle): "
  f"{'PASS — collapses to chance, no leakage.' if leak else 'FAIL — investigate leakage.'}")

wr = ns["within-record shuffle"]["roc_auc"]
share = (wr - 0.5) / (REAL_ROC - 0.5) if REAL_ROC > 0.5 else float("nan")
p(f"  DIAGNOSTIC (within-record shuffle): ROC-AUC={wr:.4f}")
p(f"    -> ~{share:.0%} of the model's skill above chance is attributable to")
p(f"       between-record differences (who the subject is) rather than to")
p(f"       discriminating apnoeic from normal minutes WITHIN a subject.")
pd.DataFrame(sh_rows).to_csv(R / "shuffle_test.csv", index=False)

# 6b. seed variance
p("\n6b. SEED VARIANCE — re-run with alternate seeds.")
sv = []
for sd in [42, 7, 2024]:
    t0 = time.time()
    o = run_xgb(X, y, g, sd, shuffle_seed=sd)
    sv.append(dict(seed=sd, pr_auc=average_precision_score(y, o),
                   roc_auc=roc_auc_score(y, o),
                   recall=recall_score(y, (o >= .5).astype(int)),
                   accuracy=accuracy_score(y, (o >= .5).astype(int))))
    p(f"  seed={sd:5d} PR-AUC={sv[-1]['pr_auc']:.4f} ROC-AUC={sv[-1]['roc_auc']:.4f} "
      f"recall={sv[-1]['recall']:.4f} ({time.time()-t0:.0f}s)")
svd = pd.DataFrame(sv); svd.to_csv(R / "seed_variance.csv", index=False)
p(f"  PR-AUC  mean={svd.pr_auc.mean():.4f} sd={svd.pr_auc.std():.4f} "
  f"range={svd.pr_auc.max()-svd.pr_auc.min():.4f}")
p(f"  ROC-AUC mean={svd.roc_auc.mean():.4f} sd={svd.roc_auc.std():.4f}")

# 6c. failure analysis
p("\n6c. PER-RECORD FAILURE ANALYSIS")
prd = pd.read_csv(R / "per_record_metrics.csv")
worst = prd.nsmallest(4, "accuracy")
p("  worst records:")
p(worst[["record", "class_prefix", "apnea_rate", "recall", "accuracy", "roc_auc"]]
  .round(3).to_string(index=False))
r1, p1 = stats.spearmanr(prd.apnea_rate, prd.accuracy)
p(f"\n  accuracy vs record apnoea rate: Spearman r={r1:+.3f} p={p1:.4f}")
r2, p2 = stats.spearmanr(prd.dropna(subset=["roc_auc"]).apnea_rate,
                         prd.dropna(subset=["roc_auc"]).roc_auc)
p(f"  ROC-AUC  vs record apnoea rate: Spearman r={r2:+.3f} p={p2:.4f}")
p("  -> a strong negative accuracy correlation means the model's global 0.5")
p("     threshold is mis-set for records whose apnoea rate is far from 50%.")

(R / "stage5_6_report.txt").write_text("\n".join(rep), encoding="utf-8")
print("\nwrote stage5_6_report.txt")
