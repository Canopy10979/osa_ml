"""Cross-dataset comparison: does per-minute apnoea detection transfer?

Two datasets in this repo pose the same task at the same unit of observation --
one minute of ECG-derived heart-rate variability, scored apnoea / normal by an
expert -- and both featurise a 5-minute window centred on the target minute:

  A = dataset_apnea_ecg  (PhysioNet Apnea-ECG, 27 subjects, 48.9% apnoea)
  H = dataset_apnea_hrv  (HuGCDN2014,          77 subjects, 22.1% apnoea)

They were featurised by independent pipelines but share 39 identically-defined
columns, so a model trained on one can be scored on the other. That is the
strongest available test of whether the findings are real physiology or
dataset-specific artefact: transfer performance cannot be inflated by any
leakage internal to a single dataset.

Outputs -> cross_dataset/results/
  transfer_metrics.csv      train-on-one / test-on-other, both directions
  feature_agreement.csv     per-feature univariate AUC in each dataset
"""
from pathlib import Path
import numpy as np, pandas as pd, json, warnings
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cross_dataset" / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

A = pd.read_parquet(ROOT / "dataset_apnea_ecg" / "structured" / "minute_features.parquet")
H = pd.read_parquet(ROOT / "dataset_apnea_hrv" / "structured" / "minute_features.parquet")

fa = set(json.load(open(ROOT / "dataset_apnea_ecg" / "models" / "params.json"))["features"])
fh = set(json.load(open(ROOT / "dataset_apnea_hrv" / "models" / "params.json"))["features"])
SHARED = sorted(fa & fh)

A = A.rename(columns={"record": "subject"})

# A feature that is CONSTANT in one dataset is unusable for transfer, and
# actively dangerous: StandardScaler fitted on the constant side assigns it
# scale 1.0, so the other dataset's real values arrive unnormalised and saturate
# the model. That is exactly what happened here -- dataset_apnea_ecg's entire
# CVHR family is identically 0.0 (see below), which drove every transferred
# prediction to 0.0 and produced a spurious ROC-AUC of exactly 0.500.
DEGENERATE = sorted(c for c in SHARED if A[c].nunique() <= 1 or H[c].nunique() <= 1)
if DEGENERATE:
    print(f"dropping {len(DEGENERATE)} features constant in one dataset: {DEGENERATE}")
    print("  (dataset_apnea_ecg's band_power() returns 0.0 unless >2 FFT bins fall in "
          "the band;\n   at 4 Hz / nperseg=256 the 0.010-0.040 Hz CVHR band contains "
          "exactly 2, so it\n   always returned 0. The CVHR features there are a bug, "
          "not a null result.)")
    SHARED = [c for c in SHARED if c not in DEGENERATE]

print(f"shared features: {len(SHARED)}")
print(f"A dataset_apnea_ecg: {len(A):,} minutes, {A.subject.nunique()} subjects, "
      f"{A.apnea.mean():.1%} apnoea")
print(f"H dataset_apnea_hrv: {len(H):,} minutes, {H.subject.nunique()} subjects, "
      f"{H.apnea.mean():.1%} apnoea")


def mat(df):
    """Feature matrix, standardised WITHIN its own dataset.

    The two pipelines share column names but not units: dataset_apnea_ecg keeps
    RR in seconds (median rr_mean 0.916) while dataset_apnea_hrv keeps it in
    milliseconds (median 903), with band powers differing by ~1e6 in
    consequence. Feeding one cohort's raw values to a model scaled on the other
    saturates it -- that, plus the constant CVHR columns, is why the first
    attempt returned a prediction of exactly 0.0 for every row.

    Standardising each cohort against its own mean and s.d. removes both the
    unit mismatch and any baseline offset between cohorts. It uses no labels, so
    it leaks nothing; what survives is the *shape* of each feature's relationship
    to apnoea, which is the thing actually being tested for transfer.
    """
    Z = df[SHARED].astype(float)
    Z = (Z - Z.median()) / Z.std(ddof=0).replace(0, np.nan)
    return Z.to_numpy(), df["apnea"].to_numpy()


def model():
    # Logistic regression only. Transfer across cohorts with different apnoea
    # prevalence and different recording hardware is exactly where a heavily
    # tuned tree ensemble overfits its source; the linear model is the fair and
    # interpretable choice, and it was the recall leader on both datasets.
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=5000, C=0.5,
                                              class_weight="balanced",
                                              random_state=SEED))])


rows = []
for name_tr, dtr, name_te, dte in [("apnea_hrv", H, "apnea_ecg", A),
                                   ("apnea_ecg", A, "apnea_hrv", H)]:
    Xtr, ytr = mat(dtr)
    Xte, yte = mat(dte)
    m = model().fit(Xtr, ytr)
    pr = m.predict_proba(Xte)[:, 1]
    base = yte.mean()
    rows.append(dict(train=name_tr, test=name_te, n_train=len(ytr), n_test=len(yte),
                     test_base_rate=base,
                     roc_auc=roc_auc_score(yte, pr),
                     pr_auc=average_precision_score(yte, pr),
                     pr_auc_lift=average_precision_score(yte, pr) / base,
                     recall=recall_score(yte, (pr >= 0.5).astype(int), zero_division=0)))
    print(f"\ntrain {name_tr} -> test {name_te}: "
          f"ROC-AUC={rows[-1]['roc_auc']:.4f}  PR-AUC={rows[-1]['pr_auc']:.4f} "
          f"(base {base:.4f}, lift {rows[-1]['pr_auc_lift']:.2f}x)")

    # within-dataset reference on the SAME shared feature set, so the comparison
    # isolates transfer loss rather than feature-set differences
    Xs, ys = mat(dte)
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    o = np.zeros(len(ys))
    for tr, te in cv.split(Xs, ys, dte.subject.to_numpy()):
        o[te] = model().fit(Xs[tr], ys[tr]).predict_proba(Xs[te])[:, 1]
    rows.append(dict(train=f"{name_te} (within, shared feats)", test=name_te,
                     n_train=len(ys), n_test=len(ys), test_base_rate=base,
                     roc_auc=roc_auc_score(ys, o),
                     pr_auc=average_precision_score(ys, o),
                     pr_auc_lift=average_precision_score(ys, o) / base,
                     recall=recall_score(ys, (o >= 0.5).astype(int), zero_division=0)))
    print(f"  within-{name_te} reference:      "
          f"ROC-AUC={rows[-1]['roc_auc']:.4f}  PR-AUC={rows[-1]['pr_auc']:.4f} "
          f"(lift {rows[-1]['pr_auc_lift']:.2f}x)")

pd.DataFrame(rows).to_csv(OUT / "transfer_metrics.csv", index=False)

# ---- do the same features matter in both datasets? ----
def uni(df):
    out = {}
    for c in SHARED:
        v = df[c]; m = v.notna()
        if m.sum() < 100 or v[m].nunique() < 2:
            continue
        out[c] = roc_auc_score(df.apnea[m], v[m])
    return pd.Series(out)

ua, uh = uni(A), uni(H)
fa_df = pd.DataFrame({"feature": SHARED}).set_index("feature")
fa_df["auc_apnea_ecg"] = ua
fa_df["auc_apnea_hrv"] = uh
# fold both onto the same side of 0.5 so "informative" is comparable
fa_df["signed_ecg"] = fa_df.auc_apnea_ecg
fa_df["signed_hrv"] = fa_df.auc_apnea_hrv
fa_df["abs_ecg"] = (fa_df.auc_apnea_ecg - 0.5).abs() + 0.5
fa_df["abs_hrv"] = (fa_df.auc_apnea_hrv - 0.5).abs() + 0.5
fa_df = fa_df.dropna(subset=["abs_ecg", "abs_hrv"]).sort_values("abs_hrv", ascending=False)
fa_df.to_csv(OUT / "feature_agreement.csv")

rho = stats.spearmanr(fa_df.abs_ecg, fa_df.abs_hrv)
rdir = stats.pearsonr(fa_df.signed_ecg - 0.5, fa_df.signed_hrv - 0.5)
print(f"\nfeature-importance agreement across datasets:")
print(f"  Spearman rank r = {rho[0]:.3f} (p={rho[1]:.2g})  on |AUC-0.5|")
print(f"  direction agreement (signed) Pearson r = {rdir[0]:.3f} (p={rdir[1]:.2g})")
same_dir = ((fa_df.signed_ecg - 0.5) * (fa_df.signed_hrv - 0.5) > 0).mean()
print(f"  features pointing the SAME direction in both: {same_dir:.0%}")
print("\ntop shared features by dataset_apnea_hrv AUC:")
print(fa_df[["abs_hrv", "abs_ecg"]].head(10).round(3).to_string())
print(f"\n[write] {OUT/'transfer_metrics.csv'}")
print(f"[write] {OUT/'feature_agreement.csv'}")
