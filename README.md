# GROUP34_CENG313 — Jigsaw: Community Rule Violation Prediction

This project predicts the probability that a Reddit comment violates a given community rule. The target is `rule_violation`: `1` indicates a violation and `0` indicates no violation.

## Group 34

| Full name | Student ID | GitHub username |
|---|---|---|
| Hacı Kerem Mermer | 23118080769 | Keremmermer40 |
| Necdet Atakan Aydın | 24118080793 | necdetatakanaydin |

[Competition](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) · [Course instructions](https://github.com/ceng313-2026)

## Data and setup

Accept the competition rules on Kaggle, download `train.csv`, `test.csv`, and `sample_submission.csv` from the **Data** tab, and place them in the local `data/` directory. Do not commit the dataset. The training data used here contains **2,029 rows**, while the sample test data contains **10 rows**. The sample test data is unlabeled and does not represent the full hidden competition test set.

Run the following commands from the project root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter notebook assignment1_eda_baseline.ipynb
```

On Linux/macOS, use `.venv/bin/python` as the Python executable. Run the notebook from beginning to end and save its plots and tables. The submitted notebook includes saved outputs.

**Google Colab:** Upload and open `assignment1_eda_baseline.ipynb`. When prompted by the first cell, upload `GROUP34_CENG313_teslim.zip`. When prompted by the data-loading cell, upload all three CSV files together. The first code cell extracts only the shared source files and requirements from the package. A CPU runtime is sufficient. After execution, use **File → Download → Download .ipynb** to save the notebook with its outputs.

## Repository structure

```text
assignment1_eda_baseline.ipynb   # Executed EDA, baseline, and held-out evaluation
requirements.txt
.gitignore
src/
  __init__.py
  data.py                      # Data loading, text preparation, and splits
  models.py                    # Models and development-set comparison
  utils.py                     # Evaluation helpers
  eda.py                       # Exploratory plots
submissions/
  heldout_metrics.csv
  heldout_per_rule.csv
  development_results.csv
  heldout_score.png
  README.md
train_baseline.py              # Earlier exploratory script using three seeds
```

The notebook contains the main submission and final evaluation. `train_baseline.py` reproduces the earlier exploratory comparison; it does not generate the separate held-out report. The Assignment 2 notebook will be prepared for the second submission.

## Method and results

The rule and comment are combined into a single text input. Missing text values are replaced with empty strings. We compare a Dummy classifier, word TF-IDF with Logistic Regression, and character TF-IDF with Logistic Regression. TF-IDF is fitted only on the corresponding training partition. Detailed parameters and preprocessing decisions are documented in the notebook and `src/models.py`.

Competition closed — evaluated on a held-out split.

**No official Kaggle score is available.** Late-submission access has not been verified through the team's account. Since labeled hidden test data is unavailable, we report the following local held-out result; it is not an official competition score.

| Model | Held-out rows | Mean per-rule AUC | ROC-AUC | Accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| char_logistic | 406 | 0.8210 | 0.8199 | 0.7414 | 0.7586 |

A fixed 20% test partition is created with `random_state=42`, stratified by the target. We remove 107 rows from the training side whose normalized comment text also appears in the test partition, leaving 1,516 development rows. Model selection uses grouped validation splits within the development partition only.

Earlier project versions explored the entire dataset, so this test partition cannot be considered historically untouched. In the revised workflow, the held-out score is not used for model selection. This evaluation does not measure generalization to unseen rules.

[Score visualization](submissions/heldout_score.png) · [Metrics](submissions/heldout_metrics.csv) · [Per-rule results](submissions/heldout_per_rule.csv)

### Comparison with published competition results

The [fifth-place solution report](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/5th-place-solution-diverse-ensemble) reports a **public leaderboard score of 0.92084** for one LLaMA component. The numerical difference from our local result is **0.0999 AUC points**. This component score is not the team's final ensemble score or the winning score.

The winner's [first-place solution report](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution) reports a final ensemble **public score of 0.9344** and **private score of 0.9293**. The numerical difference between its public score and our local mean per-rule AUC is **0.1134 points**.

These comparisons use different evaluation datasets and therefore do not establish a direct performance gap on the same test set. Only published scores were used for comparison; no public notebook code or cells were copied.

## Account and submission steps still required

- Accept the organization invitation and transfer the repository to `ceng313-2026/GROUP34_CENG313`.
- Verify that both members have Write access and meaningful contribution commits.
- Remove the old `src/__pycache__/` directory from the current GitHub file tree.
- If late submission is accessible, add the official score and a screenshot. The current report uses local evaluation because labeled hidden test data is unavailable.
- After updating all files, tag the final submission commit as `assignment1`.

**Assignment 1 deadline:** November 13, 2026, 23:59. **Assignment 2 deadline:** December 30, 2026, 23:59. Organization membership, repository tags, and contribution history cannot be verified from this file package.
