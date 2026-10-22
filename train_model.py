import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from liver_utils import COLS, RANDOM_STATE, build_model


def train(csv_path="indian_liver_patient.csv", out_path="model.joblib"):
    df = pd.read_csv(csv_path).drop_duplicates().reset_index(drop=True)
    df["Disease"] = (df["Dataset"] == 1).astype(int)          # 1 = liver disease
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    X, y = df[COLS], df["Disease"]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    model = build_model()

    # Pick threshold on out-of-fold TRAIN predictions: best specificity with recall >= 0.80
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    best_t, best_spec = 0.5, -1
    for t in np.arange(0.30, 0.86, 0.02):
        pred = (oof >= t).astype(int)
        tn, fp, _, _ = confusion_matrix(y_tr, pred).ravel()
        if recall_score(y_tr, pred) >= 0.80 and tn / (tn + fp) > best_spec:
            best_t, best_spec = float(round(t, 2)), tn / (tn + fp)

    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= best_t).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "recall": recall_score(y_te, pred),
        "roc_auc": roc_auc_score(y_te, proba),
    }
    print("Threshold:", best_t, "| Test metrics:", {k: round(v, 3) for k, v in metrics.items()})

    model.fit(X, y)   # refit on all data for the deployed model
    joblib.dump({"model": model, "threshold": best_t, "metrics": metrics}, out_path)
    print("Saved", out_path)
    return out_path


if __name__ == "__main__":
    train()
