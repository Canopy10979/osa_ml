# Person-Level OSA Aggregation Proof

This document records the exact repository evidence showing that the OSA analysis performs **person-level aggregation**, not only individual apnea-event detection.

## Verified Pipelines

Person-level aggregation is explicitly implemented in:

- `common/apnea_hrv_validate.py`
- `common/ucddb_models.py`

Both pipelines first produce event-level predictions, then group those predictions by subject, calculate a duration-normalized predicted apnea/event burden, and finally generate one OSA / no-OSA prediction for each subject.

---

## Apnea-HRV: Minute-Level Predictions to Person-Level OSA

### Step 1: Group minute predictions by subject

```python
sp = (oofdf.assign(pred=(ob >= 0.5).astype(int))
      .groupby("subject").agg(
          n_min=("y", "size"),
          pred_apnea_min=("pred", "sum"),
          true_apnea_min=("y", "sum")
      ).reset_index())
```

The `groupby("subject")` call is the direct proof that predictions are pooled at the person level.

### Step 2: Convert predictions into a subject-level predicted apnea index

```python
sp["hours"] = sp.n_min / 60
sp["ai_pred"] = sp.pred_apnea_min / sp.hours
sp["ai_true"] = sp.true_apnea_min / sp.hours
```

This creates a duration-normalized predicted apnea burden for every subject.

### Step 3: Attach each person's OSA outcome

```python
sp = sp.merge(
    sub[["subject", "group", "osa", "split"]],
    on="subject"
)
```

### Step 4: Produce one OSA prediction per person

```python
sp["osa_pred"] = (sp.ai_pred >= thr).astype(int)
```

The script then evaluates person-level accuracy, balanced accuracy, sensitivity, specificity, ROC-AUC, and a confusion matrix.

Final subject-level predictions are saved to:

```text
dataset_apnea_hrv/results/subject_level_predictions.csv
```

### Validation detail

The script states that the per-minute predictions are out-of-fold and that each subject is held out when that subject's minutes are predicted. This prevents a person from contributing training examples to the model that generates that same person's aggregated predictions.

---

## UCDDB: Epoch-Level Predictions to Person-Level OSA

The UCDDB model script explicitly defines two separate questions:

```text
per-epoch  -- is a respiratory event happening in this 30 s window?
per-subject -- does this person have OSA (AHI >= 15)?
```

### Step 1: Group epoch predictions by subject

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

Again, `groupby("subject")` is the direct aggregation step.

### Step 2: Merge subject-level clinical outcomes

```python
sp = sp.merge(
    sub[[
        "subject", "sleep_hours", "ahi", "osa_15",
        "epworth", "BMI", "Age"
    ]],
    on="subject"
)
```

### Step 3: Calculate a predicted subject-level event index

```python
sp["ai_pred"] = sp.pred_ep / sp.sleep_hours
```

### Step 4: Evaluate the aggregate as a person-level OSA score

```python
auc_sub = roc_auc_score(sp.osa_15, sp.ai_pred)
```

### Step 5: Produce one binary OSA prediction per subject

```python
sp["osa_pred"] = (sp.ai_pred >= thr).astype(int)
```

The pipeline then calculates balanced accuracy, sensitivity, specificity, and a confusion matrix for person-level OSA classification.

Final predictions are saved to:

```text
dataset_ucddb_v2/results/subject_level_predictions.csv
```

The script also implements leave-one-subject-out threshold evaluation:

```python
loo_pred = []
for i in range(len(sp)):
    tr = sp.drop(index=i)
    b2 = [balanced_accuracy_score(
        tr.osa_15,
        (tr.ai_pred >= t).astype(int)
    ) for t in cands]
    loo_pred.append(
        int(sp.ai_pred.iloc[i] >= cands[int(np.argmax(b2))])
    )
```

This refits the classification threshold without using the held-out subject to select that threshold.

---

## Aggregation Logic

```text
many minute / epoch predictions
            ↓
     group by subject
            ↓
count predicted apnea/event windows
            ↓
 normalize by time asleep / recorded
            ↓
 subject-level predicted event index
            ↓
 compare with subject-level threshold
            ↓
        OSA / No OSA
```

## Interpretation

The repository therefore contains two analysis layers:

1. **Event detection** — whether an apnea or respiratory event occurs in an individual time window.
2. **Person-level OSA classification** — whether the aggregate predicted event burden for an entire subject indicates OSA.

This supports describing the project as a system that moves from individual physiological events to subject-level OSA prediction.

## Limitation / Scientific Wording

The model-derived `ai_pred` values should not automatically be described as clinically interchangeable with polysomnographic AHI.

- Apnea-HRV uses a predicted apnea-minutes-per-hour index and evaluates it against subject-level OSA status.
- UCDDB evaluates its predicted event index against recorded AHI and the binary `osa_15` target.
- Threshold optimization can be optimistic when fitted on the same available subjects, which is why the UCDDB script also reports threshold-free ROC-AUC and leave-one-subject-out threshold evaluation.

The safest wording is:

> Minute- or epoch-level event predictions were aggregated by subject into a duration-normalized predicted apnea/event index, which was then used to produce one person-level OSA classification per participant.

## Conclusion

Yes: the analysis contains explicit and reproducible person-level aggregation. The final output is not limited to individual apnea events; the code generates one OSA / no-OSA prediction per subject from aggregated event predictions.
