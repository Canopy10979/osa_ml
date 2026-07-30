# dataset_ucddb — report

**Status: placeholder.** This file is required by the layout in `osa-ml-skill.md`
but no per-dataset report has been written yet.

The outputs currently in this repo could not be attributed to a single raw
dataset — `results/` held four pipeline *run variants*
(`regenerated`, `batch_sampled`, `balanced_models`, `block_cross_validation`),
not per-dataset results. Those were placed in `../cross_dataset/` and
`../validation/` rather than being split arbitrarily between dataset folders.

To populate this report, re-run the pipeline scoped to `raw/` and write
`structured/`, `models/` and `results/` for this dataset alone.
