"""
Module   : Predictive Classification Model — Spam Email Detector
Author   : Himanshu
Purpose  : Implements a supervised machine learning pipeline using scikit-learn
           for binary text classification (spam vs. ham) with full evaluation metrics.
Dataset  : Synthetic Spam Email dataset (reproducible via random seed)
Standards: PEP 8 compliant | Modular cell-based structure for Jupyter Notebook
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────

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

# Reproducibility seed — fixes random state across all stochastic operations
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — DATA INGESTION & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Generates a reproducible synthetic spam/ham dataset and executes
    feature engineering: text normalization and label encoding to
    produce a clean feature matrix (X) and binary target vector (y).

    Returns
    -------
    X : pd.Series
        Raw email text corpus (feature input for vectorization).
    y : np.ndarray
        Integer-encoded target labels — 1: Spam, 0: Ham.
    """
    # --- Synthetic corpus construction ---
    # Each tuple: (email_text, label) where label ∈ {spam, ham}
    spam_samples = [
        ("Win a free iPhone now! Click here to claim your prize",         "spam"),
        ("Congratulations! You've been selected for a $1000 reward",      "spam"),
        ("Urgent: Your bank account has been compromised, act now",        "spam"),
        ("Buy cheap meds online no prescription needed",                   "spam"),
        ("You are a lucky winner of our lottery jackpot",                  "spam"),
        ("Free vacation to Bahamas, limited time offer, click now",        "spam"),
        ("Make money fast from home, guaranteed $5000 per week",           "spam"),
        ("Exclusive deal: 90% off luxury watches today only",              "spam"),
        ("Your PayPal account needs immediate verification",               "spam"),
        ("Hot singles in your area want to meet you tonight",              "spam"),
        ("Earn unlimited income working from home click here",             "spam"),
        ("Special promotion: buy one get ten free, click to claim",        "spam"),
        ("Alert: suspicious login detected, reset your password now",      "spam"),
        ("Nigerian prince needs your help to transfer funds",              "spam"),
        ("You have been pre-approved for a $50,000 loan",                  "spam"),
        ("Miracle weight loss pill, lose 30 pounds in 30 days",            "spam"),
        ("Your subscription expires today, renew to avoid charges",        "spam"),
        ("Double your investment in 24 hours guaranteed returns",          "spam"),
        ("Claim your free gift card worth $500 today",                     "spam"),
        ("Virus detected on your PC, call this number immediately",        "spam"),
    ]
    ham_samples = [
        ("Hey, are we still meeting for lunch tomorrow at noon?",          "ham"),
        ("Please find the project report attached for your review",        "ham"),
        ("Can you send me the notes from today's lecture?",                "ham"),
        ("Team standup is rescheduled to 3 PM today",                      "ham"),
        ("Happy birthday! Hope you have a wonderful day",                  "ham"),
        ("The quarterly budget review is scheduled for Friday morning",    "ham"),
        ("Just checking in to see how you are doing",                      "ham"),
        ("The new software update has been deployed to production",        "ham"),
        ("Please review and approve the pull request when you get time",   "ham"),
        ("Mom asked if you're coming home this weekend",                   "ham"),
        ("Your interview is confirmed for Monday at 10 AM",                "ham"),
        ("Reminder: submit your timesheet before end of day",              "ham"),
        ("The dataset preprocessing is complete, results look promising",  "ham"),
        ("Can we move tomorrow's call to Thursday instead?",               "ham"),
        ("I've updated the documentation as per your feedback",            "ham"),
        ("Dinner reservation confirmed for two at 7:30 PM",                "ham"),
        ("The client approved the final design mockups",                   "ham"),
        ("Please upload your assignment by Sunday 11:59 PM",               "ham"),
        ("Your order has been shipped and will arrive in 3 days",          "ham"),
        ("Great work on the presentation today, very impressive",          "ham"),
    ]

    raw_data = spam_samples + ham_samples
    np.random.seed(RANDOM_STATE)
    np.random.shuffle(raw_data)  # Shuffle to break class ordering bias

    df = pd.DataFrame(raw_data, columns=["email_text", "label"])

    # --- Null value audit ---
    # Validating data integrity: zero null values expected in synthetic corpus
    null_counts = df.isnull().sum()
    print("── Null Value Audit ──────────────────────────────")
    print(null_counts)

    # --- Label encoding: spam→1, ham→0 ---
    # Executing binary label transformation via LabelEncoder (alphabetical sort: ham=0, spam=1)
    le = LabelEncoder()
    y = le.fit_transform(df["label"])   # Shape: (n_samples,) — integer target vector

    X = df["email_text"]                # Shape: (n_samples,) — raw text corpus

    print(f"\n── Dataset Shape ─────────────────────────────────")
    print(f"Samples : {len(X)}")
    print(f"Classes : {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Class distribution:\n{pd.Series(y).value_counts().rename({0: 'Ham', 1: 'Spam'})}")

    return X, y, le


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    X: pd.Series,
    y: np.ndarray,
    test_size: float = 0.2,
) -> tuple:
    """
    Partitions the corpus into training (80%) and validation (20%) sets
    using stratified sampling to preserve class distribution across splits.

    Parameters
    ----------
    X         : Raw text feature series
    y         : Encoded target vector
    test_size : Proportion allocated to validation set (default: 0.20)

    Returns
    -------
    X_train, X_test, y_train, y_test : Stratified text/label splits
    """
    # Stratify=y ensures proportional class representation in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,            # Prevents class imbalance leakage across splits
    )

    print("── Train / Test Split ────────────────────────────")
    print(f"Training samples   : {len(X_train)} ({100 * (1 - test_size):.0f}%)")
    print(f"Validation samples : {len(X_test)}  ({100 * test_size:.0f}%)")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — MODEL TRAINING (PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────

def build_and_train_model(
    X_train: pd.Series,
    y_train: np.ndarray,
) -> Pipeline:
    """
    Constructs a scikit-learn Pipeline encapsulating:
      1. CountVectorizer  — transforms raw text into a sparse BoW term-frequency matrix
      2. RandomForestClassifier — ensemble learner on the resulting feature matrix

    Pipeline design avoids data leakage: vectorizer fit occurs only on training data.

    Parameters
    ----------
    X_train : Training text corpus
    y_train : Training label vector

    Returns
    -------
    pipeline : Fitted sklearn Pipeline object
    """
    # --- Pipeline construction ---
    # CountVectorizer: converts corpus → sparse matrix of token counts (n_samples × vocabulary)
    # RandomForest: 200 decision trees via bootstrap aggregation (bagging) on token features
    pipeline = Pipeline([
        ("vectorizer", CountVectorizer(
            stop_words="english",   # Removes high-frequency low-information tokens
            ngram_range=(1, 2),     # Unigrams + bigrams: captures phrase-level patterns
            max_features=500,       # Caps vocabulary at top-500 most frequent terms
        )),
        ("classifier", RandomForestClassifier(
            n_estimators=200,       # 200 trees → stable variance reduction via bagging
            max_depth=None,         # Fully grown trees; regularized by min_samples_split
            min_samples_split=2,    # Minimum node split threshold
            random_state=RANDOM_STATE,
            n_jobs=-1,              # Parallelizes training across all CPU cores
        )),
    ])

    # Instantiating RandomForest classifier and fitting to training data matrix
    # Pipeline internally: fit_transform(vectorizer) → fit(classifier)
    pipeline.fit(X_train, y_train)

    print("── Model Training Complete ───────────────────────")
    print(f"Vectorizer vocabulary size : {len(pipeline['vectorizer'].vocabulary_)} tokens")
    print(f"Classifier                 : {pipeline['classifier'].__class__.__name__}")
    print(f"Number of estimators       : {pipeline['classifier'].n_estimators}")

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    pipeline: Pipeline,
    X: pd.Series,
    y: np.ndarray,
    n_splits: int = 5,
) -> None:
    """
    Applies Stratified K-Fold cross-validation (k=5) across the full dataset
    to generate an unbiased estimate of generalization performance.

    Stratified folds preserve class ratio in each fold — critical for
    imbalanced or small datasets to prevent misleading CV accuracy.

    Parameters
    ----------
    pipeline : Fitted sklearn Pipeline
    X        : Full text corpus
    y        : Full encoded target vector
    n_splits : Number of CV folds (default: 5)
    """
    # StratifiedKFold: maintains class proportion per fold, unlike standard KFold
    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    cv_scores = cross_val_score(
        pipeline, X, y,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1,
    )

    print("── Cross-Validation Results (Stratified K-Fold) ──")
    print(f"Fold accuracies : {np.round(cv_scores, 4)}")
    print(f"Mean accuracy   : {cv_scores.mean():.4f}")
    print(f"Std deviation   : {cv_scores.std():.4f}  (lower = more stable)")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: np.ndarray,
    class_names: list[str],
) -> None:
    """
    Generates classification metrics (Precision, Recall, F1-Score) and
    confusion matrix to quantify predictive accuracy on held-out test data.

    Metrics computed:
    - Accuracy     : Global correct prediction rate
    - Precision    : TP / (TP + FP) — minimizes false spam flags on ham
    - Recall       : TP / (TP + FN) — minimizes missed spam detection
    - F1-Score     : Harmonic mean of Precision and Recall
    - Confusion Matrix : 2×2 error breakdown (TP, TN, FP, FN)

    Parameters
    ----------
    pipeline     : Fitted sklearn Pipeline
    X_test       : Validation text corpus (unseen during training)
    y_test       : True labels for validation set
    class_names  : Human-readable class labels ['Ham', 'Spam']
    """
    # Inferring class probabilities → argmax → predicted label vector (n_test_samples,)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=class_names)

    print("── Evaluation Metrics ────────────────────────────")
    print(f"Test Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print("Classification Report:")
    print(report)

    # --- Confusion Matrix Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Spam Email Classifier — Model Evaluation Dashboard", fontsize=14, fontweight="bold")

    # Plot 1: Confusion Matrix Heatmap
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion Matrix\n(Counts of TP, TN, FP, FN)")

    # Plot 2: Per-Class Metric Bar Chart (Precision, Recall, F1)
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1]
    )
    metrics_df = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }, index=class_names)

    metrics_df.plot(kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452", "#55A868"],
                    edgecolor="white", width=0.6)
    axes[1].set_title("Per-Class Metrics\n(Precision | Recall | F1-Score)")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xticklabels(class_names, rotation=0)
    axes[1].legend(loc="lower right")
    axes[1].axhline(y=1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.savefig("task4_evaluation_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("── Dashboard saved → task4_evaluation_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — FEATURE IMPORTANCE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(pipeline: Pipeline, top_n: int = 20) -> None:
    """
    Extracts and visualizes top-N discriminative features (tokens) from the
    fitted RandomForest via mean impurity decrease (Gini importance).

    High Gini importance → token contributes most to reducing node impurity
    across all 200 trees — directly interpretable as spam-signal strength.

    Parameters
    ----------
    pipeline : Fitted sklearn Pipeline containing vectorizer + classifier
    top_n    : Number of top features to display (default: 20)
    """
    # Retrieve feature names from vectorizer vocabulary
    feature_names = np.array(pipeline["vectorizer"].get_feature_names_out())

    # Extract mean Gini importances across all 200 trees; shape: (n_features,)
    importances = pipeline["classifier"].feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_n]  # Descending sort → top-N indices

    top_features = feature_names[top_indices]
    top_scores = importances[top_indices]

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_features[::-1], top_scores[::-1],
                   color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Mean Gini Importance (Impurity Reduction)")
    ax.set_title(f"Top {top_n} Discriminative Features — RandomForest\n"
                 f"(Higher score = stronger spam/ham signal)")
    ax.axvline(x=np.mean(top_scores), color="red", linestyle="--",
               linewidth=1, label=f"Mean importance: {np.mean(top_scores):.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("task4_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("── Feature importance chart saved → task4_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — INFERENCE (LIVE PREDICTION DEMO)
# ─────────────────────────────────────────────────────────────────────────────

def predict_new_emails(pipeline: Pipeline, class_names: list[str]) -> None:
    """
    Demonstrates real-time inference: passes new, unseen email strings
    through the fitted pipeline and returns predicted class with confidence.

    Confidence = max class probability from RandomForest's soft voting
    across all 200 trees (predict_proba).

    Parameters
    ----------
    pipeline     : Fitted sklearn Pipeline
    class_names  : Human-readable class labels ['Ham', 'Spam']
    """
    test_emails = [
        "Congratulations! You have won a free luxury car, claim now!",
        "Hey, can you review the pull request I sent this morning?",
        "Exclusive offer: buy cheap medicines without a prescription",
        "The team meeting is rescheduled to 4 PM today, please confirm",
        "Your account has been compromised, click here to secure it now",
    ]

    print("── Live Inference Demo ───────────────────────────")
    print(f"{'Email':<60} {'Prediction':<8} {'Confidence':>10}")
    print("─" * 82)

    # predict_proba returns matrix (n_samples × n_classes); max gives confidence score
    probabilities = pipeline.predict_proba(test_emails)
    predictions = pipeline.predict(test_emails)

    for email, pred, prob in zip(test_emails, predictions, probabilities):
        label = class_names[pred]
        confidence = np.max(prob)
        flag = "🚨" if pred == 1 else "✅"
        print(f"{email[:58]:<60} {flag} {label:<6}  {confidence:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — MAIN PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrates the full supervised ML pipeline in sequential stages:
    Ingestion → Preprocessing → Split → Train → CV → Evaluate → Infer
    """
    print("=" * 52)
    print("  TASK 4 — MACHINE LEARNING MODEL IMPLEMENTATION  ")
    print("  Spam Email Binary Classifier | scikit-learn      ")
    print("=" * 52, "\n")

    # Stage 1: Data ingestion and preprocessing
    X, y, le = load_and_preprocess_data()
    class_names = list(le.classes_)                # ['ham', 'spam'] → display labels

    # Stage 2: Stratified train/test partition (80/20)
    print()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

    # Stage 3: Pipeline construction and model training
    print()
    pipeline = build_and_train_model(X_train, y_train)

    # Stage 4: Stratified K-Fold cross-validation (k=5) on full dataset
    print()
    cross_validate_model(pipeline, X, y, n_splits=5)

    # Stage 5: Evaluation on held-out test set
    print()
    evaluate_model(pipeline, X_test, y_test, class_names)

    # Stage 6: Feature importance analysis
    print()
    plot_feature_importance(pipeline, top_n=20)

    # Stage 7: Live inference on new unseen emails
    print()
    predict_new_emails(pipeline, class_names)

    print("\n" + "=" * 52)
    print("  Pipeline complete. All outputs saved.")
    print("=" * 52)


if __name__ == "__main__":
    main()
