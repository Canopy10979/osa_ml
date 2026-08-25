# Model Evaluation Correction Note

During validation of the OSA machine-learning pipeline, a target-leakage issue was identified in the structured Apnea-ECG feature dataset.

The predictor column `apnea` was perfectly correlated with the target label and was therefore removed from all affected machine-learning feature sets.

Additional constant and label-derived features were also excluded.

Affected analyses were rerun using:

- subject-independent train/test splitting
- explicit target-leakage screening
- constant-feature removal
- correlation-based leakage checks
- revalidated ROC-AUC, sensitivity, specificity, F1, balanced accuracy, and false-positive rates

All mentor-facing PNG figures were regenerated from the corrected model outputs.

Older figures should be considered superseded by the newly generated versions.
