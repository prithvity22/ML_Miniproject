# SMS Spam Classifier (Machine Learning)

A beginner-friendly ML mini-project that classifies SMS messages as **spam** or **ham** (not spam).

## What this project does
1. Loads an SMS Spam CSV file.
2. Splits the dataset into 80% training data and 20% testing data.
3. Converts text into numeric features using TF-IDF.
4. Trains a Multinomial Naive Bayes classifier.
5. Prints accuracy, precision, recall, a confusion matrix, and a classification report.
6. Tests three messages written in the code.

## 1. Requirements
- Python 3.10 or newer
- VS Code (recommended)

## 2. Set up the project
Open this folder in VS Code. Open Terminal → New Terminal and run:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Install libraries:

```bash
python -m pip install -r requirements.txt
```

## 3. Add the dataset
Download the **SMS Spam Collection** dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Extract the download. Copy the file named `SMSSpamCollection` into this project folder, then rename it to `spam.csv`.

The script supports:
- UCI `SMSSpamCollection` tab-separated format (no header), renamed to `spam.csv`
- Common Kaggle `spam.csv` format with columns `v1` and `v2`
- CSV format with `label` and `message` columns

For the UCI file, renaming it to `spam.csv` is enough; do not open and resave it in Excel.

## 4. Run the project

```bash
python spam_classifier.py
```

The terminal prints the test-set metrics, confusion matrix, and predictions for three custom messages.

## 5. Understanding the metrics
- **Accuracy:** fraction of all test messages classified correctly.
- **Precision (spam):** among messages predicted as spam, the fraction that really are spam.
- **Recall (spam):** among all actual spam messages, the fraction detected by the model.
- **Confusion matrix:** counts correct and incorrect predictions. Rows are actual classes (`ham`, `spam`); columns are predicted classes (`ham`, `spam`).

Metric values depend on the dataset and split; they are not hard-coded.

## 6. Important notes
- The code uses `random_state=42` so the split is reproducible.
- `stratify=df["label"]` helps preserve the ham/spam ratio in train and test sets.
- TF-IDF is fitted on training messages only to avoid data leakage.
- A message's prediction is a model estimate, not a guarantee.

## Troubleshooting
- **FileNotFoundError:** ensure `spam.csv` is in the same folder as `spam_classifier.py`.
- **No module named sklearn:** activate your environment and run `python -m pip install -r requirements.txt`.
- **Wrong Python command:** try `py spam_classifier.py` on Windows.
