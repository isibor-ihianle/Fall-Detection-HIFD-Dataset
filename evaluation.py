import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupShuffleSplit

from config import RANDOM_STATE, OUT_DIR
from models import MODEL_FACTORIES, MODELS_REQUIRING_SCALING


def tune_threshold(y_val, y_scores):
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.1, 0.9, 17):
        preds = (y_scores >= thr).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def get_positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        else:
            return proba[:, -1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            return 1 / (1 + np.exp(-scores))
        else:
            return scores[:, -1]
    else:
        return model.predict(X)


def plot_and_save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved confusion matrix plot to:", out_path)


def plot_and_save_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} - ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{model_name}_roc_curve.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved ROC curve plot to:", out_path)



def subject_level_cv_for_model(model_name, factory, X, y, Xy):
    subjects = np.sort(Xy["subject"].unique())
    results = []

    feature_importances = []
    feature_names = X.columns.to_list()
    scale_needed = model_name in MODELS_REQUIRING_SCALING

    all_y_true = []
    all_y_scores = []
    all_y_pred = []

    print("\n" + "#" * 80)
    print(f"RUNNING LOSO FOR MODEL: {model_name}")
    print("#" * 80)

    for test_subj in subjects:
        print("\n" + "=" * 70)
        print(f"TEST SUBJECT = {test_subj}")
        print("=" * 70)

        test_mask = (Xy["subject"] == test_subj).values
        train_mask = ~test_mask

        X_train_full = X.iloc[train_mask].reset_index(drop=True)
        y_train_full = y[train_mask]
        Xy_train_full = Xy.iloc[train_mask].reset_index(drop=True)

        X_test = X.iloc[test_mask].reset_index(drop=True)
        y_test = y[test_mask]

        if len(np.unique(y_test)) < 2:
            print("Warning: test subject has only one class; skipping metrics.")

        groups_train = Xy_train_full["file"].values
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )
        inner_train_idx, inner_val_idx = next(
            gss.split(X_train_full, y_train_full, groups=groups_train)
        )

        X_train = X_train_full.iloc[inner_train_idx].copy()
        y_train = y_train_full[inner_train_idx]
        X_val = X_train_full.iloc[inner_val_idx].copy()
        y_val = y_train_full[inner_val_idx]

        X_test_fold = X_test.copy()

        if scale_needed:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = pd.DataFrame(scaler.transform(X_train), columns=feature_names)
            X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_names)
            X_test_fold = pd.DataFrame(
                scaler.transform(X_test_fold), columns=feature_names
            )

        print(
            "Train windows:", len(X_train),
            "Val:", len(X_val),
            "Test:", len(X_test_fold),
        )

        model = factory()
        model.fit(X_train, y_train)

        if model_name == "RandomForest" and hasattr(model, "feature_importances_"):
            feature_importances.append(model.feature_importances_)

        val_scores = get_positive_scores(model, X_val)
        best_thr, best_f1 = tune_threshold(y_val, val_scores)
        print(
            f"Best threshold on VAL for subject {test_subj}: "
            f"{best_thr:.2f} (F1={best_f1:.4f})"
        )

        test_scores = get_positive_scores(model, X_test_fold)
        y_pred_test = (test_scores >= best_thr).astype(int)

        all_y_true.append(y_test)
        all_y_scores.append(test_scores)
        all_y_pred.append(y_pred_test)

        print("\n--- TEST RESULTS (raw, tuned threshold) ---")
        report = classification_report(
            y_test, y_pred_test, digits=4, zero_division=0
        )
        print(report)
        cm = confusion_matrix(y_test, y_pred_test)
        print("Confusion matrix:\n", cm)

        report_dict = classification_report(
            y_test, y_pred_test, output_dict=True, zero_division=0
        )
        results.append(
            {
                "subject": test_subj,
                "n_test": len(y_test),
                "n_fall_test": int((y_test == 1).sum()),
                "precision_fall": report_dict["1"]["precision"],
                "recall_fall": report_dict["1"]["recall"],
                "f1_fall": report_dict["1"]["f1-score"],
            }
        )

    results_df = pd.DataFrame(results)
    print("\n================ SUMMARY (LOSO, window-level) ================")
    print(results_df)
    print("\nMean performance across subjects:")
    print(results_df.mean(numeric_only=True))

    top_feats_df = None
    if model_name == "RandomForest" and feature_importances:
        importances_arr = np.vstack(feature_importances)
        mean_importances = importances_arr.mean(axis=0)
        idx = np.argsort(mean_importances)[::-1][:50]

        print(
            "\n================ TOP 50 FEATURES (mean over LOSO folds, RF) ================"
        )
        for rank, i in enumerate(idx, start=1):
            print(f"{rank:2d}. {feature_names[i]:30s} {mean_importances[i]:.6f}")

        top_feats_df = pd.DataFrame(
            {
                "feature": [feature_names[i] for i in idx],
                "importance_mean": [mean_importances[i] for i in idx],
            }
        )

    all_y_true = np.concatenate(all_y_true)
    all_y_scores = np.concatenate(all_y_scores)
    all_y_pred = np.concatenate(all_y_pred)

    print(f"\nGlobal confusion matrix for {model_name}:")
    print(confusion_matrix(all_y_true, all_y_pred))

    plot_and_save_confusion_matrix(all_y_true, all_y_pred, model_name)
    plot_and_save_roc_curve(all_y_true, all_y_scores, model_name)

    return results_df, top_feats_df


def run_all_models(X, y, Xy):
    all_model_results = {}
    rf_top_feats_df = None

    for name, factory in MODEL_FACTORIES.items():
        results_df, top_feats_df = subject_level_cv_for_model(
            name, factory, X, y, Xy
        )
        all_model_results[name] = results_df
        if name == "RandomForest":
            rf_top_feats_df = top_feats_df

    rows = []
    for name, res in all_model_results.items():
        m = res.mean(numeric_only=True)
        rows.append(
            {
                "model": name,
                "mean_precision_fall": m["precision_fall"],
                "mean_recall_fall": m["recall_fall"],
                "mean_f1_fall": m["f1_fall"],
                "mean_n_test": m["n_test"],
                "mean_n_fall_test": m["n_fall_test"],
            }
        )
    comparison_df = pd.DataFrame(rows).sort_values(
        by="mean_f1_fall", ascending=False
    )

    print(
        "\n================ MODEL COMPARISON (LOSO, fall class) ================"
    )
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(
        os.path.join(OUT_DIR, "model_comparison_loso.csv"), index=False
    )
    if rf_top_feats_df is not None:
        rf_top_feats_df.to_csv(
            os.path.join(OUT_DIR, "rf_top50_features_loso.csv"), index=False
        )

    return comparison_df, rf_top_feats_df
