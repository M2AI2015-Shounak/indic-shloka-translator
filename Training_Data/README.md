# Training_Data

Optional parallel material for the custom model (`notebooks/02_train_custom_model.ipynb`).
Put CSV / XLSX / JSON / JSONL files here with exactly these columns:

| Column | Content |
|---|---|
| `Source_Text` | Indic-language text |
| `Language` | Language name (`Sanskrit`) or code (`san_Deva`) — one of the 21 supported languages |
| `English` | Reference English translation |

Only use material you have the right to train on. Pairs that match (exactly or ≥ 90 % similar) any
shloka in `reference/` or `Input_Folder/` are removed automatically (leakage guard).
Files here are git-ignored.
