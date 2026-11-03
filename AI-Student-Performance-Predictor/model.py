"""Data generation and machine-learning functions for the mini-project."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "assignment_score",
    "sleep_hours",
    "extracurricular_hours",
]


def create_dataset(n_samples=600, random_state=42):
    """Create a reproducible synthetic dataset for demonstration and learning."""
    rng = np.random.default_rng(random_state)
    study = rng.uniform(0, 10, n_samples)
    attendance = rng.uniform(40, 100, n_samples)
    previous = rng.uniform(20, 100, n_samples)
    assignments = rng.uniform(25, 100, n_samples)
    sleep = rng.uniform(4, 10, n_samples)
    activities = rng.uniform(0, 15, n_samples)

    # A transparent synthetic relationship with noise; not a real-world rule.
    score = (
        2.8 * study
        + 0.20 * attendance
        + 0.22 * previous
        + 0.16 * assignments
        + 1.2 * sleep
        - 0.35 * activities
        + rng.normal(0, 9, n_samples)
    )
    passed = (score >= 43).astype(int)

    return pd.DataFrame({
        "study_hours": study.round(1),
        "attendance": attendance.round(1),
        "previous_score": previous.round(1),
        "assignment_score": assignments.round(1),
        "sleep_hours": sleep.round(1),
        "extracurricular_hours": activities.round(1),
        "result": passed,
    })


def train_model(data):
    """Train a Random Forest and return model plus held-out test metrics."""
    X = data[FEATURES]
    y = data["result"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=160, max_depth=8, min_samples_leaf=3, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "report": classification_report(
            y_test, predictions, target_names=["Needs Support", "Pass"], zero_division=0
        ),
    }
    return model, metrics, X_test, y_test


def predict_student(model, student):
    """Return a class label and estimated pass probability for one student profile."""
    row = pd.DataFrame([student], columns=FEATURES)
    probability = float(model.predict_proba(row)[0][1])
    return {
        "pass_probability": probability,
        "label": "Pass" if probability >= 0.5 else "Needs Support",
    }
