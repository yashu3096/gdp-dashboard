# AI-Based Student Performance Prediction System

**Branch:** Computer Science & Engineering — Artificial Intelligence and Machine Learning (AIML)  
**Project type:** Mini Project  
**Technology:** Python, Machine Learning, Streamlit

## Overview

This project demonstrates a machine-learning workflow that estimates whether a student profile is more likely to **Pass** or **Need Support**. A Random Forest Classifier is trained using study hours, attendance, previous exam score, assignment average, sleep hours, and extracurricular hours.

The Streamlit dashboard provides:
- An interactive form to enter student details and estimate pass probability.
- A preview and visual exploration of the dataset.
- Held-out test metrics: accuracy, precision, recall, and classification report.
- Random Forest feature-importance visualization.

## Run locally

Python 3.9 or newer is recommended.

```bash
cd AI-Student-Performance-Predictor
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies and start the app:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually http://localhost:8501).

## Machine-learning workflow

1. Generate a reproducible synthetic dataset (600 rows).
2. Separate six input features from the binary target.
3. Split data into training and test sets using stratification.
4. Train a Random Forest Classifier.
5. Evaluate on the held-out test set.
6. Use the trained model to estimate the pass probability for a user-entered profile.

## Dataset and responsible use

The project intentionally generates **synthetic teaching data** in code; it does not contain real student records. The relationship used to create labels is illustrative and includes random noise. Therefore, test metrics only describe performance on this synthetic dataset. They are not evidence of real-world accuracy.

Do not use this demo to make actual grading, admissions, scholarship, employment, or disciplinary decisions. A real deployment would require consent, representative and validated data, bias testing, privacy safeguards, and human oversight.

## Project structure

```text
AI-Student-Performance-Predictor/
├── app.py          # Streamlit user interface
├── model.py        # Dataset generation, training, and prediction
├── requirements.txt
└── README.md
```

## Future scope

- Evaluate on a consented, anonymized, real educational dataset.
- Compare Random Forest with Logistic Regression, Decision Tree, and Gradient Boosting.
- Add model explainability with SHAP.
- Add secure teacher login and privacy-preserving storage.
- Track model drift and retrain only after proper validation.

## Author

AIML — Computer Science & Engineering Mini Project.
