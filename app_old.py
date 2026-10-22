
import os

import joblib
import pandas as pd
import streamlit as st

from liver_utils import COLS

st.set_page_config(page_title="Liver Disease Prediction", page_icon="🩺", layout="centered")


@st.cache_resource
def load_bundle():
    # Train on first run if model.joblib is missing or was saved with an incompatible scikit-learn
    try:
        return joblib.load("model.joblib")
    except Exception:
        from train_model import train
        train()
        return joblib.load("model.joblib")


bundle = load_bundle()
model, threshold, metrics = bundle["model"], bundle["threshold"], bundle["metrics"]

st.title("🩺 Liver Disease Prediction System")
st.caption("Gradient Boosting + AdaBoost ensemble · Indian Liver Patient Dataset")
st.warning("Decision-support tool for educational use only. Not a medical device. "
           "Always confirm with clinical evaluation.")

with st.form("patient_form"):
    st.subheader("Patient details")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", 1, 100, 45)
    gender = c2.selectbox("Gender", ["Female", "Male"], index=1)

    st.subheader("Lab results")
    c1, c2 = st.columns(2)
    tb = c1.number_input("Total bilirubin (mg/dL)", 0.0, 80.0, 1.0, 0.1)
    db = c2.number_input("Direct bilirubin (mg/dL)", 0.0, 20.0, 0.3, 0.1)
    alp = c1.number_input("Alkaline phosphatase (IU/L)", 0, 2500, 200, 10)
    alt = c2.number_input("ALT / Alamine aminotransferase (IU/L)", 0, 2500, 30, 5)
    ast = c1.number_input("AST / Aspartate aminotransferase (IU/L)", 0, 5000, 35, 5)
    tp = c2.number_input("Total proteins (g/dL)", 0.0, 10.0, 6.5, 0.1)
    alb = c1.number_input("Albumin (g/dL)", 0.0, 6.0, 3.2, 0.1)
    agr = c2.number_input("Albumin/Globulin ratio", 0.0, 3.0, 1.0, 0.05)
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([[age, 1 if gender == "Male" else 0, tb, db, alp, alt, ast, tp, alb, agr]], columns=COLS)
    p = float(model.predict_proba(row)[0, 1])
    st.subheader("Result")
    if p >= threshold:
        st.error(f"HIGH RISK: liver disease likely (probability {p:.0%})")
    else:
        st.success(f"LOW RISK: liver disease unlikely (probability {p:.0%})")
    st.progress(min(max(p, 0.0), 1.0))
    st.caption(f"Decision threshold: {threshold:.2f}")

with st.expander("Model performance (held-out test set)"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Recall", f"{metrics['recall']:.1%}")
    c3.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")
    st.caption("Small dataset (583 patients); expect roughly 70-75% accuracy. "
               "The threshold favours recall so sick patients are not missed, at the cost of false alarms.")
