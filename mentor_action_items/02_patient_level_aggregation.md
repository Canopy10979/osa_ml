# Patient-Level Aggregation Summary

## Research Question

Does the analysis aggregate individual apnea-event predictions into a **person-level prediction of whether a subject has OSA**, rather than stopping at minute- or epoch-level event detection?

## Short Answer

**Yes.** The repository contains explicit subject-level aggregation in both the Apnea-HRV and UCDDB pipelines. In each case, out-of-fold event predictions are grouped by subject, converted into a duration-normalized predicted apnea/event index, and then used to generate a single OSA / no-OSA prediction for each person.

This means the analysis answers two distinct questions:

1. **Event level:** Is an apnea/respiratory event occurring in this minute or epoch?
2. **Person level:** After aggregating all predictions for one subject, does that person meet the OSA classification target?

---

## 1. Apnea-HRV Person-Level Aggregation

**Source:** `common/apnea_hrv_validate.py`

The validation script explicitly states that its clinically useful output is to estimate each subject's apnea index from per-minute predictions and classify OSA.

### Aggregation code

```python
sp = (oofdf.assign(pred=(ob >= 0.5).astype(int))
      .groupby("subject").agg(
          n_min=("y", "size"),
          pred_apnea_min=("pred", "sum"),
          true_apnea_min=("y", "sum")
      ).reset_index())
```

This converts many minute-level predictions into **one row per subject**.

The predicted apnea burden is then normalized by recording duration:

```python
sp["hours"] = sp.n_min / 60
sp["ai_pred"] = sp.pred_apnea_min / sp.hours
sp["ai_true"] = sp.true_apnea_min / sp.hours
```

The person-level clinical label is merged in:

```python
sp = sp.merge(
    sub[["subject", "group", "osa", "split"]],
    on="subject"
)
```

Finally, the predicted index is converted into a binary person-level OSA prediction:

```python
sp["osa_pred"] = (sp.ai_pred >= thr).astype(int)
```

The resulting person-level predictions are evaluated with accuracy, balanced accuracy, sensitivity, specificity, ROC-AUC, and a confusion matrix, and are saved to:

`dataset_apnea_hrv/results/subject_level_predictions.csv`

### Leakage / validation safeguard

The Apnea-HRV script states that the per-minute predictions used for aggregation are **out-of-fold** and grouped by subject, so each person's minutes are predicted while that subject is held out from model training.

---

## 2. UCDDB Person-Level Aggregation

**Source:** `common/ucddb_models.py`

The UCDDB pipeline explicitly separates two tasks:

- per-epoch respiratory-event detection; and
- per-subject OSA classification, defined in the script as **AHI >= 15**.

### Aggregation code

```python
sp = (pd.DataFrame({
        "subject": g,
        "pred": (ob >= .5).astype(int),
        "y": y
      })
      .groupby("subject").agg(
          n_ep=("y", "size"),
          pred_ep=("pred", "sum"),
          true_ep=("y", "sum")
      ).reset_index())
```

This again reduces many epoch predictions to **one record per person**.

Subject-level clinical information is then merged:

```python
sp = sp.merge(
    sub[[
        "subject", "sleep_hours", "ahi", "osa_15",
        "epworth", "BMI", "Age"
    ]],
    on="subject"
)
```

The duration-normalized predicted event index is computed as:

```python
sp["ai_pred"] = sp.pred_ep / sp.sleep_hours
```

The index is evaluated against recorded AHI and against the binary person-level target `osa_15`:

```python
auc_sub = roc_auc_score(sp.osa_15, sp.ai_pred)
```

A single binary OSA prediction is then created for every subject:

```python
sp["osa_pred"] = (sp.ai_pred >= thr).astype(int)
```

The file also reports balanced accuracy, sensitivity, specificity, and a confusion matrix, and saves the final table to:

`dataset_ucddb_v2/results/subject_level_predictions.csv`

The UCDDB pipeline additionally performs leave-one-subject-out threshold evaluation so that the held-out subject does not determine its own operating threshold.

---

## 3. Aggregation Formula

The core logic can be summarized as:

```text
individual event predictions
        ↓
group predictions by subject
        ↓
count predicted apnea / respiratory-event windows
        ↓
divide by recording or sleep duration
        ↓
predicted apnea/event index for that person
        ↓
apply a person-level decision threshold
        ↓
OSA or no OSA
```

For Apnea-HRV:

```text
ai_pred = predicted apnea minutes / recording hours
```

For UCDDB:

```text
ai_pred = predicted event epochs / sleep hours
```

The exact quantities differ because the underlying datasets use different windowing schemes, but both pipelines perform the same conceptual transition from event-level predictions to one person-level score and one person-level OSA classification.

---

## 4. What This Proves

The project does **not** use individual event predictions as its only final outcome. It contains a second analysis layer that pools those predictions within each subject and asks whether the overall predicted respiratory-event burden separates subjects with OSA from subjects without OSA.

Therefore, it is accurate to describe the project as containing both:

- apnea/respiratory-event detection; and
- person-level OSA prediction through subject-level aggregation.

---

## 5. Important Limitations

The aggregated prediction should not automatically be described as a clinically validated AHI replacement.

- In Apnea-HRV, the person-level operating threshold is optimized on the available subjects, although the script also reports performance on the official `T` subset separately.
- In UCDDB, the script explicitly labels the same-subject threshold optimization as optimistic and treats threshold-free subject-level ROC-AUC as the more defensible summary; it also performs leave-one-subject-out threshold evaluation.
- The person-level output is therefore best described as a **model-derived apnea/event burden or predicted index used for OSA classification**, not as a replacement for clinical polysomnographic AHI.

---

## Conclusion

The repository contains explicit, reproducible person-level aggregation. Minute- or epoch-level predictions are grouped by subject, normalized by duration, and converted into one binary OSA prediction per person. This directly supports the research goal of moving beyond individual apnea-event detection toward determining whether a person is likely to have OSA.
