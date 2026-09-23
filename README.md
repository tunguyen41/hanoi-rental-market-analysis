# Hanoi rental market analysis

Exploratory analysis of asking rents from Nhatot and Alonhadat. The observation unit is a **listing**, not a transaction or a verified unique property.

## Run the EDA

From the project directory, using Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_eda.py
```

If `.venv` already exists, install the requirements and run the final command. The runner executes both notebooks sequentially in fresh kernels using the calling Python interpreter, then saves their outputs. Alternatively, open the notebooks in VS Code, select the `.venv` kernel and Run All in order.

1. [01_data_audit.ipynb](notebooks/01_data_audit.ipynb): audit, review decisions, derived variables and analysis-set eligibility.
2. [02_rental_eda.ipynb](notebooks/02_rental_eda.ipynb): 10 figures, source comparisons, sensitivity analysis and modeling readiness.

Input: `data/processed/listings.csv`. The EDA preserves this file and does not crawl new data. In the current snapshot, `scraped_at` records parser execution, not a verified download date.

## Outputs

- [EDA report](reports/eda_report.md): findings, sample sizes, limitations and modeling preparation.
- [Data dictionary](reports/data_dictionary.md): original and derived fields and their processing rules.
- `reports/figures/`: 10 standalone PNG figures.
- `data/analysis/listings_analysis.csv`: all input rows, quality flags, review decisions and eligibility fields.
- `data/analysis/review_queue.csv`: records requiring review and saved decisions.

Notebook 02 only reads the finalized analysis dataset. If a new issue is found, update the review queue and rerun both notebooks.

## Review workflow

Set `scope_decision`, `price_decision` and `area_decision` to `keep`, `exclude`, `unresolved`, or leave them blank. Keep/exclude decisions require `review_note` and `reviewer`. Do not edit original context fields in the queue. Decisions persist across runs and are rejected if a listing's title/price_raw/area_raw has changed.

Initial decisions were made by Codex from titles and CSV fields, **without verifying the original webpages**. Unresolved scope/price cases are excluded from the main price set; missing or unclear area alone does not remove monthly rent. Keywords flag records rather than prove errors. Private houses are excluded from the primary price/m² set. Shared-bed/per-person rents are not pooled with whole-room rents. Repost groups remain candidates, used in sensitivity checks and conservatively in proposed group splitting.

Groups below 10 show counts only; groups of 10–29 are exploratory; groups of at least 30 support main comparisons but do not guarantee representativeness. When fully matched district groups are too small for bedroom/furnishing comparisons, charts relax district matching and state the limitation. EDA v1 does not train a model or infer temporal trends.

## Language and source data

Code documentation, messages, notebooks, figures, review notes and reports use English. Original Vietnamese listing text, place names, category codes and raw furnishing values are preserved for traceability. Vietnamese parsing patterns and test fixtures remain necessary to process those source records. Derived furnishing labels and property-type presentation labels use English.

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Checks cover price/m² units, feature-specific missingness, review persistence, original-data preservation, group splitting and per-source comparison thresholds.

The existing `.gitignore` contains `*.md`, so new Markdown reports are ignored by Git even though they are generated and readable locally. Add specific files when you want to version them; the ignore rule has not been changed. The already tracked README still records changes.
