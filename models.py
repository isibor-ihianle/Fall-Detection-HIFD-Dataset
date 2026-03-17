from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE

MODELS_REQUIRING_SCALING = {"KNN", "LogReg", "SVM", "XGBoost", "RandomForest", "LightGBM"}


def make_rf():
    return RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        n_estimators=300,
        max_depth=None,
    )


def make_xgb():
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_lgbm():
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_knn():
    return KNeighborsClassifier(
        n_neighbors=15,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )


def make_logreg():
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
    )


def make_svm():
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


MODEL_FACTORIES = {
    "RandomForest": make_rf,
    "XGBoost": make_xgb,
    "LightGBM": make_lgbm,
    "KNN": make_knn,
    "LogReg": make_logreg,
    "SVM": make_svm,
}
