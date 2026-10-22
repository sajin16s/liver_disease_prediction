"""Streamlit app with MySQL storage.  Run:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st
from mysql.connector import Error as MySQLError

import db
from liver_utils import COLS

st.set_page_config(page_title="Liver Disease Prediction", page_icon="🩺", layout="wide")


@st.cache_resource
def load_bundle():
    # Train on first run if model.joblib is missing or was saved with an incompatible scikit-learn
    try:
        return joblib.load("model.joblib")
    except Exception:
        from train_model import train
        train()
        return joblib.load("model.joblib")


@st.cache_resource
def setup_database():
    """Create the database/tables once per server run. Returns (ok, error_message)."""
    try:
        db.init_db()
        return True, ""
    except (MySQLError, ValueError) as e:
        return False, str(e)


bundle = load_bundle()
model, threshold, metrics = bundle["model"], bundle["threshold"], bundle["metrics"]
db_ok, db_error = setup_database()

st.title("🩺 Liver Disease Prediction System")
st.caption("Gradient Boosting + AdaBoost ensemble · Indian Liver Patient Dataset · MySQL storage")
st.warning("Decision-support tool for educational use only. Not a medical device. "
           "Do not enter real patient identities; use anonymous IDs or dummy names.")

if db_ok:
    st.sidebar.success("MySQL connected")
else:
    st.sidebar.error("MySQL not connected")
    st.sidebar.caption(f"{db_error}")
    st.sidebar.caption("Predictions still work, but results will not be saved. "
                       "Check your MySQL server and credentials in .streamlit/secrets.toml.")

with st.sidebar.expander("Model performance (test set)"):
    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    st.metric("Recall", f"{metrics['recall']:.1%}")
    st.metric("ROC-AUC", f"{metrics['roc_auc']:.2f}")

tab_predict, tab_history = st.tabs(["🔬 New prediction", "🗂️ Patient history"])

# ---------------------------------------------------------------- Predict
with tab_predict:
    with st.form("patient_form"):
        st.subheader("Patient details")
        c1, c2, c3 = st.columns([2, 1, 1])
        name = c1.text_input("Patient name / ID", placeholder="e.g. Patient-001")
        age = c2.number_input("Age (years)", 1, 100, 45)
        gender = c3.selectbox("Gender", ["Female", "Male"], index=1)

        st.subheader("Lab results")
        c1, c2 = st.columns(2)
        tb = c1.number_input("Total bilirubin (mg/dL)", 0.0, 80.0, 1.0, 0.1)
        db_ = c2.number_input("Direct bilirubin (mg/dL)", 0.0, 20.0, 0.3, 0.1)
        alp = c1.number_input("Alkaline phosphatase (IU/L)", 0, 2500, 200, 10)
        alt = c2.number_input("ALT / Alamine aminotransferase (IU/L)", 0, 2500, 30, 5)
        ast = c1.number_input("AST / Aspartate aminotransferase (IU/L)", 0, 5000, 35, 5)
        tp = c2.number_input("Total proteins (g/dL)", 0.0, 10.0, 6.5, 0.1)
        alb = c1.number_input("Albumin (g/dL)", 0.0, 6.0, 3.2, 0.1)
        agr = c2.number_input("Albumin/Globulin ratio", 0.0, 3.0, 1.0, 0.05)
        submitted = st.form_submit_button("Predict & save", type="primary", width="stretch")

    if submitted:
        if not name.strip():
            st.error("Please enter a patient name or ID.")
        else:
            labs = [tb, db_, alp, alt, ast, tp, alb, agr]
            row = pd.DataFrame([[age, 1 if gender == "Male" else 0, *labs]], columns=COLS)
            p = float(model.predict_proba(row)[0, 1])
            risk = "HIGH" if p >= threshold else "LOW"

            st.subheader("Result")
            if risk == "HIGH":
                st.error(f"HIGH RISK: liver disease likely (probability {p:.0%})")
            else:
                st.success(f"LOW RISK: liver disease unlikely (probability {p:.0%})")
            st.progress(min(max(p, 0.0), 1.0))
            st.caption(f"Decision threshold: {threshold:.2f}")

            if db_ok:
                try:
                    pid = db.save_prediction(name, age, gender, labs, p, risk, threshold)
                    st.info(f"Saved to MySQL (record #{pid}).")
                except MySQLError as e:
                    st.warning(f"Prediction done, but saving to MySQL failed: {e}")
            else:
                st.warning("Prediction done, but MySQL is not connected, so nothing was saved.")

# ---------------------------------------------------------------- History
with tab_history:
    if not db_ok:
        st.info("Connect MySQL to view saved predictions.")
    else:
        try:
            stats = db.get_stats()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total predictions", stats["total"])
            m2.metric("High-risk cases", stats["high"])
            m3.metric("Average probability", f"{stats['avg_prob']:.0%}")

            search = st.text_input("Search by patient name", placeholder="Type to filter...")
            history = db.fetch_history(search)
            if history.empty:
                st.info("No records yet. Make a prediction first.")
            else:
                st.dataframe(history, width="stretch", hide_index=True)
                st.download_button("Download as CSV", history.to_csv(index=False).encode("utf-8"),
                                   "prediction_history.csv", "text/csv")

                with st.expander("Delete a record"):
                    del_id = st.selectbox("Record ID", history["ID"].tolist())
                    if st.button("Delete selected record"):
                        if db.delete_prediction(del_id):
                            st.success(f"Deleted record #{del_id}.")
                            st.rerun()
        except MySQLError as e:
            st.error(f"Database error: {e}")
