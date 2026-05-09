import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Keep the seed for consistent results
RANDOM_STATE = 42

#1. Data Ingestion & Preprocessing
def load_and_preprocess_data():
    """
    Creates a dummy dataset of spam and ham emails, and encodes the labels to 1 and 0.
    """
    # TODO: Replace this synthetic data with a real Kaggle dataset (like SMS Spam Collection) if I have more time.
    spam_samples = [
        ("Win a free iPhone now! Click here to claim your prize", "spam"),
        ("Congratulations! You've been selected for a $1000 reward", "spam"),
        ("Urgent: Your bank account has been compromised, act now", "spam"),
        ("Buy cheap meds online no prescription needed", "spam"),
        ("You are a lucky winner of our lottery jackpot", "spam"),
        ("Free vacation to Bahamas, limited time offer, click now", "spam"),
        ("Make money fast from home, guaranteed $5000 per week", "spam"),
        ("Exclusive deal: 90% off luxury watches today only", "spam"),
        ("Your PayPal account needs immediate verification", "spam"),
        ("Hot singles in your area want to meet you tonight", "spam"),
        ("Earn unlimited income working from home click here", "spam"),
        ("Special promotion: buy one get ten free, click to claim", "spam"),
        ("Alert: suspicious login detected, reset your password now", "spam"),
        ("Nigerian prince needs your help to transfer funds", "spam"),
        ("You have been pre-approved for a $50,000 loan", "spam"),
        ("Miracle weight loss pill, lose 30 pounds in 30 days", "spam"),
        ("Your subscription expires today, renew to avoid charges", "spam"),
        ("Double your investment in 24 hours guaranteed returns", "spam"),
        ("Claim your free gift card worth $500 today", "spam"),
        ("Virus detected on your PC, call this number immediately", "spam"),
    ]
    ham_samples = [
        ("Hey, are we still meeting for lunch tomorrow at noon?", "ham"),
        ("Please find the project report attached for your review", "ham"),
        ("Can you send me the notes from today's lecture?", "ham"),
        ("Team standup is rescheduled to 3 PM today", "ham"),
        ("Happy birthday! Hope you have a wonderful day", "ham"),
        ("The quarterly budget review is scheduled for Friday morning", "ham"),
        ("Just checking in to see how you are doing", "ham"),
        ("The new software update has been deployed to production", "ham"),
        ("Please review and approve the pull request when you get time", "ham"),
        ("Mom asked if you're coming home this weekend", "ham"),
        ("Your interview is confirmed for Monday at 10 AM", "ham"),
        ("Reminder: submit your timesheet before end of day", "ham"),
        ("The dataset preprocessing is complete, results look promising", "ham"),
        ("Can we move tomorrow's call to Thursday instead?", "ham"),
        ("I've updated the documentation as per your feedback", "ham"),
        ("Dinner reservation confirmed for two at 7:30 PM", "ham"),
        ("The client approved the final design mockups", "ham"),
        ("Please upload your assignment by Sunday 11:59 PM", "ham"),
        ("Your order has been shipped and will arrive in 3 days", "ham"),
        ("Great work on the presentation today, very impressive", "ham"),
    ]

    raw_data = spam_samples + ham_samples
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(raw_data)

    df = pd.DataFrame(raw_data, columns=["email_text", "label"])

    # Check for missing values
    print("Checking for nulls:")
    print(df.isnull().sum())

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df["email_text"]

    print(f"\nDataset Stats:")
    print(f"Total Samples: {len(X)}")
    print(f"Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return X, y, le

### 2. Model Training and Evaluation

def build_and_train_model(X_train, y_train):
    """
    Sets up the pipeline with a vectorizer and a Random Forest classifier.
    """
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=500)),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def predict_new_emails(pipeline, class_names):
    test_emails = [
        "Congratulations! You have won a free luxury car, claim now!",
        "Hey, can you review the pull request I sent this morning?",
        "Exclusive offer: buy cheap medicines without a prescription",
        "The team meeting is rescheduled to 4 PM today, please confirm",
        "Your account has been compromised, click here to secure it now",
    ]

    print("\nTesting with some fake emails:")
    probabilities = pipeline.predict_proba(test_emails)
    predictions = pipeline.predict(test_emails)

    for email, pred, prob in zip(test_emails, predictions, probabilities):
        label = class_names[pred]
        conf = np.max(prob)
        print(f"{label.upper()} ({conf:.1%}): {email[:50]}...")
