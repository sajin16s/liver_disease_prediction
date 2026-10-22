
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
COLS = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
    "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_and_Globulin_Ratio",
]
SKEWED = ["Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
          "Alamine_Aminotransferase", "Aspartate_Aminotransferase"]


def engineer(X):
    X = pd.DataFrame(X, columns=COLS).copy()
    X["AST_ALT_Ratio"] = X["Aspartate_Aminotransferase"] / (X["Alamine_Aminotransferase"] + 1e-6)
    X["Direct_Total_Bili_Ratio"] = X["Direct_Bilirubin"] / (X["Total_Bilirubin"] + 1e-6)
    for c in SKEWED + ["AST_ALT_Ratio"]:
        X[c] = np.log1p(X[c])
    return X


def build_model():
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                    random_state=RANDOM_STATE)
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),
                             n_estimators=50, learning_rate=0.5, random_state=RANDOM_STATE)
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("engineer", FunctionTransformer(engineer)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(mutual_info_classif, k=11)),
        ("model", VotingClassifier([("gb", gb), ("ada", ada)], voting="soft")),
    ])
