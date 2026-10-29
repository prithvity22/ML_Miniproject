"""
SMS Spam Classifier
- Loads an SMS spam CSV
- Splits data 80:20
- Converts text to TF-IDF features
- Trains a Multinomial Naive Bayes model
- Reports accuracy, precision, recall, and confusion matrix
- Tests three custom messages
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

DATA_PATH = Path(__file__).parent / "spam.csv"

def load_dataset(path=DATA_PATH):
    """Load common SMS Spam Collection CSV formats."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put the SMS spam CSV in this project folder and name it spam.csv."
        )

    # Try typical CSV formats, including Kaggle's v1/v2 and UCI tab-separated file.
    try:
        df = pd.read_csv(path, encoding="latin-1")
    except Exception:
        df = pd.read_csv(path, sep="\\t", header=None, encoding="latin-1")

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Common Kaggle dataset: v1 = label, v2 = message; extra columns are often empty.
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
    elif "label" in df.columns and "message" in df.columns:
        pass
    elif "category" in df.columns and "message" in df.columns:
        df = df.rename(columns={"category": "label"})
    else:
        # UCI SMS Spam Collection commonly has two columns and no header.
        df = pd.read_csv(path, sep="\\t", header=None, encoding="latin-1")
        if df.shape[1] < 2:
            # Also support a comma-separated two-column file without recognizable headers.
            df = pd.read_csv(path, header=None, encoding="latin-1")
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]

    df = df[["label", "message"]].dropna()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"] != ""]
    if df.empty:
        raise ValueError("No usable rows found. Ensure labels are 'ham' or 'spam' and messages are present.")
    return df

def main():
    df = load_dataset()
    print("=" * 60)
    print("SMS SPAM CLASSIFIER")
    print("=" * 60)
    print(f"Dataset rows used: {len(df)}")
    print("Label counts:")
    print(df["label"].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    # Convert messages to numerical TF-IDF features. Fit only on training text.
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print("\\n--- Test-set evaluation ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, pos_label='spam', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, pos_label='spam', zero_division=0):.4f}")
    print("\\nConfusion matrix (rows = actual [ham, spam], columns = predicted [ham, spam]):")
    labels = ["ham", "spam"]
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels),
        index=["Actual ham", "Actual spam"],
        columns=["Predicted ham", "Predicted spam"]
    ))
    print("\\nClassification report:")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

    custom_messages = [
        "Congratulations! You have won a free prize. Call now to claim it!",
        "Hey, are we still meeting at the library at 4 pm?",
        "URGENT! Your account has been selected for a cash reward. Click this link now."
    ]
    print("\\n--- Testing 3 custom messages ---")
    custom_tfidf = vectorizer.transform(custom_messages)
    custom_predictions = model.predict(custom_tfidf)
    for i, (message, prediction) in enumerate(zip(custom_messages, custom_predictions), start=1):
        print(f"\\n{i}. Message: {message}")
        print(f"   Prediction: {prediction.upper()}")

if __name__ == "__main__":
    main()
