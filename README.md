# Jigsaw Community Rule Violation Prediction

**Course:** CENG313 — Introduction to Data Science  
**Group:** 34

We investigate whether a Reddit comment violates a specified community rule. Our project combines exploratory data analysis with simple text classification models and evaluates their predictions on a local held-out split.

## Team

| Name | Student ID | GitHub |
|---|---|---|
| Hacı Kerem Mermer | 23118080769 | [Keremmermer40](https://github.com/Keremmermer40) |
| Necdet Atakan Aydın | 24118080793 | [necdetatakanaydin](https://github.com/necdetatakanaydin) |

## Problem and Data

We use the [Jigsaw — Agile Community Rules Classification](https://www.kaggle.com/competitions/jigsaw-agile-community-rules) dataset. Each prediction concerns a comment and a particular rule, rather than the comment's general toxicity.

- **Target:** `rule_violation`, where `1` indicates a violation and `0` indicates compliance.
- **Training data:** 2,029 labeled rows covering two rules.
- **Available fields:** comment text, rule text, subreddit, positive and negative examples, and a row identifier.
- **Sample test data:** 10 unlabeled rows; this is not the full hidden competition test set.

The notebook examines missing values, duplicate comments, class balance, rule and subreddit distributions, and comment lengths. Each exploratory plot is accompanied by an interpretation.

To obtain the data, accept the competition rules on Kaggle and download the files from the **Data** tab. Place `train.csv`, `test.csv`, and `sample_submission.csv` in a local `data/` folder. Dataset files are excluded from version control.

## Baseline Approach

We combine the rule and comment as `RULE: <rule> COMMENT: <comment>`. Missing text is replaced with an empty string. The baseline does not use row identifiers, subreddit names, or example comments as model features.

Three models are compared:

| Model | Representation |
|---|---|
| Dummy classifier | Class-prior reference |
| Word TF-IDF + Logistic Regression | Word n-grams of length 1–2 |
| Character TF-IDF + Logistic Regression | Character n-grams of length 3–5 using `char_wb` |

Both TF-IDF models use `min_df=2` and up to 30,000 features. Vectorizers are fitted only on the training partition of each split. Detailed preprocessing decisions and model parameters are documented in the notebook and shared source files.

## Evaluation and Results

Competition closed — evaluated on a held-out split.

We reserve 20% of the labeled data using a stratified split with `random_state=42`. To prevent the same normalized comment from appearing on both sides, 107 overlapping rows are removed from the development partition. This leaves **1,516 development rows** and **406 held-out rows**.

Model selection uses grouped validation within the development partition with seeds 7, 21, and 42. The selected character TF-IDF model is then trained on the development partition and evaluated on the held-out data.

| Selected model | Mean per-rule ROC-AUC | Overall ROC-AUC | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Character TF-IDF + Logistic Regression | **0.8210** | 0.8199 | 0.7414 | 0.7586 |

The primary local metric is the equally weighted mean of the ROC-AUC values for the two rules. Accuracy and F1 use a probability threshold of 0.5.

**These are local evaluation results, not official Kaggle public or private scores.** No official Kaggle submission score is currently reported. Earlier exploratory work used the full labeled dataset, so the held-out partition is not historically untouched. The evaluation also does not establish generalization to unseen rules.

Result files are available in [`submissions/`](submissions/):

- [`heldout_metrics.csv`](submissions/heldout_metrics.csv): held-out metrics and split sizes.
- [`heldout_per_rule.csv`](submissions/heldout_per_rule.csv): rule-level AUC values.
- [`development_results.csv`](submissions/development_results.csv): development validation results.
- [`heldout_score.png`](submissions/heldout_score.png): a visualization of the local results.

## Project Layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── assignment1_eda_baseline.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── utils.py
│   └── eda.py
├── submissions/
│   ├── README.md
│   ├── development_results.csv
│   ├── heldout_metrics.csv
│   ├── heldout_per_rule.csv
│   └── heldout_score.png
└── train_baseline.py
```

The notebook contains the analysis, explanations, plots, and final evaluation. Reusable functions are imported from `src/`. `train_baseline.py` retains the earlier exploratory comparison; it does not generate the notebook's separate held-out report.

## Running the Project

### Local execution

Download and extract the repository, then place the three CSV files in `data/`. From the project root, run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter notebook assignment1_eda_baseline.ipynb
```

On Linux/macOS, use `python3` to create the environment and `.venv/bin/python` for the remaining commands.

Run all notebook cells in order. The notebook includes saved outputs and can also be read without execution. Running it regenerates the evaluation files in `submissions/` and creates `outputs/submission.csv` for the available sample test rows.

### Google Colab

1. Download the repository using **Code → Download ZIP** and extract it on your computer.
2. Upload and open `assignment1_eda_baseline.ipynb` in Colab.
3. When the setup cell prompts you, upload the downloaded repository ZIP. The setup accepts both GitHub ZIPs and the provided project package.
4. When the data-loading cell prompts you, upload `train.csv`, `test.csv`, and `sample_submission.csv` together.
5. Run the remaining cells using a CPU runtime, then save the notebook with its outputs using **File → Download → Download .ipynb**.

## Further Work

Future experiments will investigate the use of example comments, alternative text representations, and evaluation on unseen rules or communities. These extensions will be developed for Assignment 2.
