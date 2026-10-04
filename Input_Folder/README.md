# Input_Folder

Drop the files to translate here. Every `.csv`, `.xlsx`, `.json` and `.jsonl` file is read
(non-recursive; hidden files, `~$` Excel lock files and this README are ignored).

## Required schema — exactly these 4 columns (case-sensitive, surrounding spaces ignored, any order)

| Column | Content |
|---|---|
| `ID` | Record identifier — must be non-empty |
| `Source` | Work the passage comes from (e.g. श्रीमद्भगवद्गीता) |
| `Verse` | Verse/passage label — kept as **text** (`1.10` stays `1.10`; `99.0` becomes `99`) |
| `Shloka` | The text to translate (may be empty → that record is `Skipped`) |

Any missing, extra, renamed or duplicated column ⇒ the whole file (or XLSX sheet) is **rejected**,
listed in the `Input_Validation` sheet with the exact difference, and the other files still run.

* **CSV** — UTF-8 (with/without BOM) or UTF-16; delimiter `,` `;` tab or `|` is auto-detected.
* **XLSX** — every non-empty sheet is validated on its own (`file::sheet`).
  Tip: format the `Verse` column as *Text* in Excel; numbers like `1.10` saved as numbers are
  stored as `1.1` by Excel itself and cannot be recovered (the run flags suspicious cases).
* **JSON** — a list of records `[{"ID": ..., ...}]`, or `{"records": [...]}`; **JSONL** — one record per line.

`Combined_All_Slokas.xlsx` (215 Sanskrit records) is included as the sample input.
