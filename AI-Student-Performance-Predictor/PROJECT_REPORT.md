# Mini Project Report
## AI-Based Student Performance Prediction System

### 1. Abstract
This mini project demonstrates how supervised machine learning can be used to estimate a student profile's likelihood of passing. A Random Forest Classifier learns from six illustrative academic and lifestyle features. A Streamlit interface allows users to enter a profile, view a prediction, explore the dataset, and inspect model evaluation metrics.

### 2. Problem Statement
Educators may want to identify students who could benefit from additional academic support. This project presents an educational prototype that demonstrates a classification workflow. It is not intended to make decisions about real students.

### 3. Objectives
- Generate and inspect a structured dataset.
- Train a supervised classification model.
- Predict a class and probability for a new input.
- Evaluate model performance on held-out data.
- Present results through a simple web dashboard.

### 4. Tools and Technologies
- Python
- NumPy and pandas for data handling
- scikit-learn for model training and evaluation
- Streamlit for the web interface
- Matplotlib for visualization

### 5. Methodology
The application creates 600 synthetic records with study hours, attendance, previous score, assignment average, sleep hours, and extracurricular hours. A transparent illustrative formula with random noise creates the binary target. The data is split into training and testing subsets (75% / 25%) with stratification. A Random Forest Classifier is trained on the training subset and evaluated on the held-out subset using accuracy, precision, recall, and a classification report.

### 6. System Modules
1. Data generation: creates reproducible synthetic records.
2. Model training: trains a Random Forest Classifier.
3. Prediction: calculates class probabilities for a user-entered profile.
4. Dashboard: presents inputs, predictions, dataset exploration, and metrics.

### 7. Expected Output
The dashboard displays the estimated pass probability and a predicted label ("Pass" or "Needs Support"). It also displays dataset samples, class distribution, evaluation metrics, and feature importance.

### 8. Limitations and Ethics
The dataset is synthetic and the target labels are generated from an illustrative formula. Consequently, metrics may be optimistic and do not establish real-world predictive validity. Student data is sensitive. Any real-world study would require consent, data minimization, security, representative sampling, fairness assessment, and human review. Predictions must never be treated as definitive judgments about a student.

### 9. Future Enhancements
- Compare multiple algorithms using cross-validation.
- Use a properly licensed, anonymized educational dataset.
- Add explainability and fairness checks.
- Add authentication and access controls before handling any real data.

### 10. Conclusion
The project demonstrates an end-to-end beginner-friendly AIML pipeline: data preparation, model training, evaluation, inference, and dashboard deployment. It is suitable as a classroom prototype and starting point for further experimentation.
