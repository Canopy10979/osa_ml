"""Stages 5-6 for dataset_apnea_hrv: inference, subject-level OSA, self-validation.

Stage 5: multicollinearity, model agreement, and the clinically useful output --
         estimating each subject's apnoea index from per-minute predictions and
         classifying OSA. This is the one dataset in the repo where that
         question is properly powered: 40 controls vs 37 patients.
Stage 6: label-shuffle test, seed variance, per-subject failure analysis.

Outputs -> results/: vif_top15.csv, model_agreement.csv, subject_level_predictions.csv,
shuffle_test.csv, seed_variance.csv, stage5_6_report.txt
"""
from pathlib import Path
import numpy as np, pandas as pd, warnings, time
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score,
                             balanced_accuracy_score, confusion_matrix, recall_score,
                             mean_absolute_error)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "dataset_apnea_hrv"
S, R = DS / "structured", DS / "results"
SEED = 42

rep = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); rep.append(s)

df = pd.read_parquet(S / "minute_features.parquet")
sub = pd.read_csv(S / "subject_level.csv")
oofdf = pd.read_csv(R / "oof_predictions.csv")
META = {"subject", "minute", "apnea", "osa", "group", "split"}
FEATS = [c for c in df.columns if c not in META and df[c].dtype.kind in "fi"]
X = df[FEATS].to_numpy(float); y = df["apnea"].to_numpy(); g = df["subject"].to_numpy()

m = pd.read_csv(R / "metrics.csv").set_index("model")
BEST = m.drop(index="_baseline_majority", errors="ignore")["pr_auc"].astype(float).idxmax()
REAL_PR = float(m.loc[BEST, "pr_auc"]); REAL_ROC = float(m.loc[BEST, "roc_auc"])
ob = oofdf[f"p_{BEST}"].to_numpy()

p("=" * 78); p("STAGE 5 — CORRELATION, INFERENCE, SUBJECT-LEVEL OSA"); p("=" * 78)
p(f"best model by PR-AUC: {BEST} (PR-AUC={REAL_PR:.4f}, ROC-AUC={REAL_ROC:.4f})")

# ---- multicollinearity ----
top15 = pd.read_csv(R / "feature_importance.csv").head(15)["feature"].tolist()
Z = df[top15].apply(pd.to_numeric, errors="coerce")
Z = Z.fillna(Z.median())
Zs = (Z - Z.mean()) / Z.std(ddof=0)
C = np.corrcoef(Zs.to_numpy().T)
vif = np.diag(np.linalg.pinv(C))
v = pd.DataFrame({"feature": top15, "vif": vif}).sort_values("vif", ascending=False)
v.to_csv(R / "vif_top15.csv", index=False)
p("\nVIF among the 15 most important features (>10 = heavy collinearity):")
p(v.round(2).to_string(index=False))
p(f"  features with VIF > 10: {(v.vif > 10).sum()} / 15")

# ---- model agreement ----
pcols = [c for c in oofdf.columns if c.startswith("p_")]
agree = oofdf[pcols].corr()
agree.to_csv(R / "model_agreement.csv")
p("\nModel agreement (Pearson r between predicted probabilities):")
p(agree.round(3).to_string())

# ---- subject-level OSA ----
p("\n" + "-" * 78)
p("SUBJECT-LEVEL OSA — 40 controls vs 37 patients")
p("-" * 78)
p("Per-minute out-of-fold predictions are pooled per subject into an estimated")
p("apnoea index (predicted apnoea minutes per hour). Every subject was held out")
p("when its minutes were predicted, so no subject saw its own data in training.")

sp = (oofdf.assign(pred=(ob >= 0.5).astype(int))
      .groupby("subject").agg(n_min=("y", "size"), pred_apnea_min=("pred", "sum"),
                              true_apnea_min=("y", "sum")).reset_index())
sp["hours"] = sp.n_min / 60
sp["ai_pred"] = sp.pred_apnea_min / sp.hours
sp["ai_true"] = sp.true_apnea_min / sp.hours
sp = sp.merge(sub[["subject", "group", "osa", "split"]], on="subject")

r_p = stats.pearsonr(sp.ai_true, sp.ai_pred)
r_s = stats.spearmanr(sp.ai_true, sp.ai_pred)
mae = mean_absolute_error(sp.ai_true, sp.ai_pred)
p(f"\nAI estimation: Pearson r={r_p[0]:.3f} (p={r_p[1]:.2g})  "
  f"Spearman r={r_s[0]:.3f}  MAE={mae:.2f} events/h")

# Choose the operating threshold on the predicted index by maximising balanced
# accuracy -- reported honestly as fitted on these same subjects.
cands = np.arange(1, 40, 0.5)
bal = [balanced_accuracy_score(sp.osa, (sp.ai_pred >= t).astype(int)) for t in cands]
thr = float(cands[int(np.argmax(bal))])
sp["osa_pred"] = (sp.ai_pred >= thr).astype(int)
acc = accuracy_score(sp.osa, sp.osa_pred); bacc = balanced_accuracy_score(sp.osa, sp.osa_pred)
sens = recall_score(sp.osa, sp.osa_pred); spec = recall_score(1 - sp.osa, 1 - sp.osa_pred)
cm = confusion_matrix(sp.osa, sp.osa_pred)
auc_sub = roc_auc_score(sp.osa, sp.ai_pred)

p(f"\nOSA classification at predicted-index threshold {thr:g}:")
p(f"  accuracy={acc:.3f}  balanced_accuracy={bacc:.3f}")
p(f"  sensitivity={sens:.3f}  specificity={spec:.3f}")
p(f"  ROC-AUC of the predicted index as a ranking score = {auc_sub:.4f}")
p(f"  confusion (rows=true CONTROL/APNEA, cols=pred):\n{cm}")
mis = sp[sp.osa != sp.osa_pred]
p(f"  misclassified subjects: {len(mis)} -> {mis.subject.tolist()}")

# threshold-free, honest: does the T (never-trained) subset separate?
spT = sp[sp.split == "T"]
p(f"\n  On the official T subjects only (n={len(spT)}): "
  f"ROC-AUC={roc_auc_score(spT.osa, spT.ai_pred):.4f}, "
  f"balanced_acc={balanced_accuracy_score(spT.osa, (spT.ai_pred>=thr).astype(int)):.3f}")

sp.to_csv(R / "subject_level_predictions.csv", index=False)
p("\nGroup separation of the ESTIMATED index:")
p(sp.groupby("group").ai_pred.describe()[["count", "mean", "min", "max"]].round(2).to_string())

p("\n" + "=" * 78); p("STAGE 6 — VALIDATING THE ANALYSIS ITSELF"); p("=" * 78)

try:
    from xgboost import XGBClassifier
    def mk(seed):
        return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.08,
                             subsample=0.9, colsample_bytree=0.8,
                             scale_pos_weight=float((y == 0).sum() / (y == 1).sum()),
                             eval_metric="aucpr", random_state=seed, n_jobs=-1)
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    def mk(seed):
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              class_weight="balanced", random_state=seed)

def run(yy, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    o = np.zeros(len(yy))
    Xf = np.nan_to_num(X, nan=np.nanmedian(X))
    for tr, te in cv.split(Xf, yy, g):
        o[te] = mk(seed).fit(Xf[tr], yy[tr]).predict_proba(Xf[te])[:, 1]
    return o

# 6a. label shuffle
p("\n6a. LABEL-SHUFFLE TEST — performance must collapse to chance.")
rng = np.random.default_rng(SEED)
sh = []
for tag, ysh in [("global shuffle", rng.permutation(y)),
                 ("within-subject shuffle",
                  pd.Series(y).groupby(g).transform(
                      lambda s: rng.permutation(s.to_numpy())).to_numpy())]:
    o = run(ysh, SEED)
    pra = average_precision_score(ysh, o); roc = roc_auc_score(ysh, o)
    sh.append(dict(test=tag, pr_auc=pra, roc_auc=roc, base_rate=ysh.mean()))
    p(f"  {tag:24s} PR-AUC={pra:.4f} (chance={ysh.mean():.4f})  ROC-AUC={roc:.4f} (chance=0.5)")
p(f"  real labels              PR-AUC={REAL_PR:.4f}  ROC-AUC={REAL_ROC:.4f}")
pd.DataFrame(sh).to_csv(R / "shuffle_test.csv", index=False)

ns = {r["test"]: r for r in sh}
leak = abs(ns["global shuffle"]["roc_auc"] - 0.5) < 0.05
p(f"  VERDICT (global shuffle): "
  f"{'PASS — collapses to chance, no leakage.' if leak else 'FAIL — investigate leakage.'}")

# The within-subject shuffle preserves each subject's apnoea burden, so a grouped
# model that has learned "this person is a severe apnoeic" still scores above
# chance with nothing leaked. It measures how much of the skill is
# between-subject, and is reported as a diagnostic, not a pass/fail gate.
wr = ns["within-subject shuffle"]["roc_auc"]
share = (wr - 0.5) / (REAL_ROC - 0.5)
p(f"  DIAGNOSTIC (within-subject shuffle): ROC-AUC={wr:.4f}")
p(f"    -> ~{share:.0%} of skill above chance is between-subject (who the person is)")
p(f"       rather than telling apnoeic from normal minutes WITHIN a subject.")

# 6b. seed variance
p("\n6b. SEED VARIANCE")
sv = []
for sd in (42, 7, 2024):
    t0 = time.time(); o = run(y, sd)
    sv.append(dict(seed=sd, pr_auc=average_precision_score(y, o),
                   roc_auc=roc_auc_score(y, o),
                   recall=recall_score(y, (o >= .5).astype(int))))
    p(f"  seed={sd:5d} PR-AUC={sv[-1]['pr_auc']:.4f} ROC-AUC={sv[-1]['roc_auc']:.4f} "
      f"recall={sv[-1]['recall']:.4f} ({time.time()-t0:.0f}s)")
sv = pd.DataFrame(sv); sv.to_csv(R / "seed_variance.csv", index=False)
p(f"  PR-AUC  mean={sv.pr_auc.mean():.4f} sd={sv.pr_auc.std():.4f}")
p(f"  ROC-AUC mean={sv.roc_auc.mean():.4f} sd={sv.roc_auc.std():.4f}")

# 6c. per-subject failure analysis
p("\n6c. PER-SUBJECT FAILURE ANALYSIS")
psm = pd.read_csv(R / "per_subject_metrics.csv")
p(f"  per-subject accuracy: mean={psm.accuracy.mean():.3f} sd={psm.accuracy.std():.3f} "
  f"min={psm.accuracy.min():.3f}")
p("  worst 5 subjects:")
p(psm.sort_values("accuracy").head(5).round(3).to_string(index=False))
rho = stats.spearmanr(psm.apnea_rate, psm.accuracy)
p(f"  accuracy vs subject apnoea rate: Spearman r={rho[0]:.3f} p={rho[1]:.3g}")

(R / "stage5_6_report.txt").write_text("\n".join(rep), encoding="utf-8")
p(f"\nwrote {R/'stage5_6_report.txt'}")
