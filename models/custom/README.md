# models/custom

Fine-tuned checkpoints of `custom-shloka-mt` are saved here by
`notebooks/02_train_custom_model.ipynb` as `custom-shloka-mt_<YYYY-MM-DD>_<HH-MM-SS>/`, each with
`model_card.md`, `metrics.json`, `training_config.yaml` and `data_manifest.csv`.
The translator loads the newest folder (`checkpoint: latest` in `config/config.yaml`) or a named one.
Weights are git-ignored.
