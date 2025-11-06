import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Load and clean data
df = pd.read_csv("data.csv")
df = df.dropna(subset=["text", "label"])

X = df["text"].astype(str)
y = df["label"].astype(str)

# Build the model pipeline
# TF-IDF converts text into numerical features.
# Logistic Regression classifies emotions.
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # dampen frequent words
        min_df=1
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handle imbalance
        C=2.0,                    # less regularization
        random_state=42
    ))
])

# Cross-validation for stable evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro")
print(f"Cross-validation F1-macro: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}")

# Train/test split for final reporting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
pipe.fit(X_train, y_train)

# Evaluate on held-out test set
y_pred = pipe.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

# 7. Save the trained model for inference
joblib.dump(pipe, "mood_baseline.joblib")
print("\nSaved model -> mood_baseline.joblib")
