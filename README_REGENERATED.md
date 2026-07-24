# Corrected OSAML Regeneration

This version rebuilds the analysis directly from the four raw patient-50 CSV files. It first constructs complete non-overlapping 30-second records, then reproducibly shuffles the records with `random_state=42` and selects every fifth shuffled record.

Normalization is fit on the training split only. This prevents information from the test set from affecting the feature scaling. Logistic Regression, Decision Tree, and Random Forest models are then trained and evaluated on the same stratified split.

## Run in VS Code

From the `OSA ML` project root:

```powershell
python .\scripts\run_pipeline.py
```

Corrected outputs are written to:

- `data/regenerated/`
- `figures/regenerated/`
- `models/regenerated/`
- `results/regenerated/`

## Important interpretation warning

`50_sleep_stage.csv` contains sleep-stage annotations. Therefore, the current target is:

- `0`: awake (`W`)
- `1`: asleep (`N1`, `N2`, `N3`, `N4`, `R`, or `REM`)

These are not scored apnea-event labels. The generated models should be described as sleep-state classifiers unless a separate apnea-event annotation source is added.
