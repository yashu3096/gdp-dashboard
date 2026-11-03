import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import create_dataset, train_model, predict_student

st.set_page_config(page_title="AI Student Performance Predictor", page_icon="🎓", layout="wide")

st.title("🎓 AI Student Performance Predictor")
st.caption("A mini-project for Artificial Intelligence & Machine Learning — CSE")
st.write(
    "Explore how study habits and academic indicators relate to outcomes, "
    "then estimate a student's pass probability using a Random Forest model."
)

with st.sidebar:
    st.header("About this project")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Task:** Binary classification (Pass / Needs Support)")
    st.write("**Data:** Reproducible synthetic teaching dataset")
    st.info("This is an educational demonstration, not a real admissions, grading, or disciplinary tool.")

data = create_dataset()
model, metrics, X_test, y_test = train_model(data)

tab1, tab2, tab3 = st.tabs(["Predict a student", "Explore dataset", "Model evaluation"])

with tab1:
    st.subheader("Enter student details")
    c1, c2, c3 = st.columns(3)
    with c1:
        study_hours = st.slider("Daily study hours", 0.0, 12.0, 3.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 75)
    with c2:
        previous_score = st.slider("Previous exam score", 0, 100, 60)
        assignment_score = st.slider("Assignment average (%)", 0, 100, 65)
    with c3:
        sleep_hours = st.slider("Average sleep (hours)", 0.0, 12.0, 7.0, 0.5)
        extracurricular = st.slider("Extracurricular hours / week", 0, 20, 4)

    if st.button("Predict outcome", type="primary", use_container_width=True):
        result = predict_student(model, {
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_score": previous_score,
            "assignment_score": assignment_score,
            "sleep_hours": sleep_hours,
            "extracurricular_hours": extracurricular,
        })
        st.divider()
        left, right = st.columns(2)
        left.metric("Estimated pass probability", f"{result['pass_probability']:.1%}")
        right.metric("Model prediction", result["label"])
        if result["label"] == "Pass":
            st.success("The model estimates a higher likelihood of passing.")
        else:
            st.warning("The model flags this profile as potentially needing academic support.")
        st.caption("Prediction is based on synthetic training data and should not be used to judge a real student.")

with tab2:
    st.subheader("Synthetic student dataset")
    st.write("The dataset is generated locally each run with a fixed random seed, so results are reproducible.")
    st.dataframe(data.head(20), use_container_width=True)
    st.caption(f"{len(data)} rows · {len(data.columns)} columns")
    st.subheader("Pass / support distribution")
    counts = data["result"].value_counts().rename(index={1: "Pass", 0: "Needs Support"})
    st.bar_chart(counts)
    st.subheader("Study hours and previous score")
    st.scatter_chart(data, x="study_hours", y="previous_score", color="result")

with tab3:
    st.subheader("Test-set performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    m2.metric("Precision", f"{metrics['precision']:.1%}")
    m3.metric("Recall", f"{metrics['recall']:.1%}")
    st.write("**Classification report**")
    st.code(metrics["report"])
    st.write("**Feature importance**")
    importance = pd.Series(model.feature_importances_, index=model.feature_names_in_).sort_values()
    st.bar_chart(importance)
    st.caption("Metrics are measured on a held-out split of synthetic data; they do not indicate real-world predictive accuracy.")

st.divider()
st.caption("Built with Python, Streamlit, pandas, Matplotlib and scikit-learn · AIML CSE Mini Project")
