"""MySQL storage layer for the Liver Disease Prediction app.

Connection settings are read (in order) from:
  1. Streamlit secrets  (.streamlit/secrets.toml, section [mysql])
  2. Environment variables MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
All queries are parameterised (no string formatting) to prevent SQL injection.
"""
import os

import mysql.connector
import pandas as pd
from mysql.connector import Error

DEFAULTS = {"host": "localhost", "port": 3306, "user": "root", "password": "", "database": "liver_db"}


def get_config():
    cfg = dict(DEFAULTS)
    try:
        import streamlit as st
        if "mysql" in st.secrets:
            cfg.update({k: v for k, v in st.secrets["mysql"].items() if k in cfg})
            return cfg
    except Exception:
        pass
    for key in cfg:
        val = os.environ.get(f"MYSQL_{key.upper()}")
        if val:
            cfg[key] = int(val) if key == "port" else val
    return cfg


def _connect(with_database=True):
    cfg = get_config()
    params = dict(host=cfg["host"], port=cfg["port"], user=cfg["user"], password=cfg["password"],
                  connection_timeout=5)
    if with_database:
        params["database"] = cfg["database"]
    return mysql.connector.connect(**params)


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS patients (
        patient_id   INT AUTO_INCREMENT PRIMARY KEY,
        patient_name VARCHAR(100) NOT NULL,
        age          TINYINT UNSIGNED NOT NULL,
        gender       ENUM('Male','Female') NOT NULL,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_patient_name (patient_name)
    ) ENGINE=InnoDB
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        prediction_id              INT AUTO_INCREMENT PRIMARY KEY,
        patient_id                 INT NOT NULL,
        total_bilirubin            FLOAT NOT NULL,
        direct_bilirubin           FLOAT NOT NULL,
        alkaline_phosphotase       FLOAT NOT NULL,
        alamine_aminotransferase   FLOAT NOT NULL,
        aspartate_aminotransferase FLOAT NOT NULL,
        total_proteins             FLOAT NOT NULL,
        albumin                    FLOAT NOT NULL,
        albumin_globulin_ratio     FLOAT NOT NULL,
        probability                FLOAT NOT NULL,
        risk_level                 ENUM('HIGH','LOW') NOT NULL,
        threshold_used             FLOAT NOT NULL,
        predicted_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT fk_patient FOREIGN KEY (patient_id)
            REFERENCES patients(patient_id) ON DELETE CASCADE,
        INDEX idx_predicted_at (predicted_at)
    ) ENGINE=InnoDB
    """,
]


def init_db():
    """Create the database and tables if they do not exist. Raises mysql.connector.Error on failure."""
    db_name = get_config()["database"]
    if not db_name.replace("_", "").isalnum():
        raise ValueError("Database name may only contain letters, digits and underscores.")
    conn = _connect(with_database=False)
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4")
        cur.execute(f"USE `{db_name}`")
        for ddl in SCHEMA:
            cur.execute(ddl)
        conn.commit()
    finally:
        conn.close()


def save_prediction(name, age, gender, labs, probability, risk_level, threshold):
    """Insert a patient and their prediction in one transaction. Returns the new prediction_id."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO patients (patient_name, age, gender) VALUES (%s, %s, %s)",
                    (name.strip(), int(age), gender))
        patient_id = cur.lastrowid
        cur.execute(
            """INSERT INTO predictions
               (patient_id, total_bilirubin, direct_bilirubin, alkaline_phosphotase,
                alamine_aminotransferase, aspartate_aminotransferase, total_proteins,
                albumin, albumin_globulin_ratio, probability, risk_level, threshold_used)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (patient_id, *[float(v) for v in labs], float(probability), risk_level, float(threshold)))
        conn.commit()
        return cur.lastrowid
    except Error:
        conn.rollback()
        raise
    finally:
        conn.close()


def fetch_history(search="", limit=500):
    """Return prediction history (newest first) as a DataFrame, optionally filtered by patient name."""
    conn = _connect()
    try:
        sql = """SELECT pr.prediction_id AS ID, p.patient_name AS Patient, p.age AS Age, p.gender AS Gender,
                        pr.total_bilirubin AS `Total Bilirubin`, pr.direct_bilirubin AS `Direct Bilirubin`,
                        pr.alkaline_phosphotase AS ALP, pr.alamine_aminotransferase AS ALT,
                        pr.aspartate_aminotransferase AS AST, pr.total_proteins AS `Total Proteins`,
                        pr.albumin AS Albumin, pr.albumin_globulin_ratio AS `A/G Ratio`,
                        ROUND(pr.probability * 100, 1) AS `Probability %`, pr.risk_level AS Risk,
                        pr.predicted_at AS `Predicted At`
                 FROM predictions pr JOIN patients p ON p.patient_id = pr.patient_id"""
        params = []
        if search.strip():
            sql += " WHERE p.patient_name LIKE %s"
            params.append(f"%{search.strip()}%")
        sql += " ORDER BY pr.predicted_at DESC, pr.prediction_id DESC LIMIT %s"
        params.append(int(limit))
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        return pd.DataFrame(cur.fetchall())
    finally:
        conn.close()


def get_stats():
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), COALESCE(SUM(risk_level='HIGH'),0), COALESCE(AVG(probability),0) FROM predictions")
        total, high, avg = cur.fetchone()
        return {"total": int(total), "high": int(high), "avg_prob": float(avg)}
    finally:
        conn.close()


def delete_prediction(prediction_id):
    """Delete one prediction; also removes the patient row if it has no other predictions."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT patient_id FROM predictions WHERE prediction_id = %s", (int(prediction_id),))
        row = cur.fetchone()
        if not row:
            return False
        cur.execute("DELETE FROM predictions WHERE prediction_id = %s", (int(prediction_id),))
        cur.execute("""DELETE FROM patients WHERE patient_id = %s
                       AND NOT EXISTS (SELECT 1 FROM predictions WHERE patient_id = %s)""", (row[0], row[0]))
        conn.commit()
        return True
    except Error:
        conn.rollback()
        raise
    finally:
        conn.close()
