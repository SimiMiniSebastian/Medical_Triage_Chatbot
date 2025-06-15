import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

def load_data():
    """
    Load and preprocess the dataset.
    """
    path = r"C:\Users\simis\disease_dataset\data\Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    df = pd.read_csv(path)

    # Rename first column to 'diseases'
    df.rename(columns={df.columns[0]: "diseases"}, inplace=True)

    # ✅ UPDATED: Identify symptom columns (assumed to be binary 0/1)
    symptom_cols = df.columns[1:]

    # ✅ Create text of symptoms only where value == 1
    df["symptoms"] = df[symptom_cols].apply(
        lambda row: ' '.join([symptom for symptom, val in row.items() if val == 1]),
        axis=1
    )

    # ✅ Clean symptom strings
    df["symptoms"] = df["symptoms"].str.lower().str.replace(r'[^a-z,\s]', '', regex=True)
    df["symptoms"] = df["symptoms"].str.replace(r'\s+', ' ', regex=True)
    df["symptoms"] = df["symptoms"].str.strip().str.replace(r'^,+|,+$', '', regex=True)

    # Drop rows with empty or stopword-only symptoms
    df = df[df["symptoms"].str.strip().astype(bool)]

    return df[["diseases", "symptoms"]]

def train_model(df):
    """
    Train Naive Bayes model.
    """
    X = df["symptoms"]
    y = df["diseases"]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Text classification pipeline
    clf = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english')),
        ('classifier', MultinomialNB())
    ])

    # Train
    clf.fit(X, y_encoded)

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/chatbot_model.joblib")
    joblib.dump(le, "models/label_encoder.joblib")

    return clf, le
