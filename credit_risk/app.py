from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from credit_risk.pipeline import (
    ComparisonResult,
    TrainedModels,
    compare_models,
    load_polish_bankruptcy_five_year,
    prediction_explanations,
)


st.set_page_config(page_title="Corporate Credit Risk", page_icon="📊", layout="wide")


@st.cache_resource(show_spinner="Training both models...")
def train_dashboard(data_path: str) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    ComparisonResult,
    TrainedModels,
    dict[str, np.ndarray],
]:
    X, y, feature_names = load_polish_bankruptcy_five_year(data_path=Path(data_path))
    comparison, models, holdout = compare_models(X, y, feature_names)
    return X, y, feature_names, comparison, models, holdout


def metric_chart(y_test: np.ndarray, predictions: dict[str, np.ndarray], chart_type: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 4))
    for model_name, probabilities in predictions.items():
        label = model_name.replace("_", " ").title()
        if chart_type == "pr":
            precision, recall, _ = precision_recall_curve(y_test, probabilities)
            axis.plot(recall, precision, label=label)
            axis.set_xlabel("Recall")
            axis.set_ylabel("Precision")
            axis.set_title("Precision-recall curve")
        elif chart_type == "roc":
            false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
            axis.plot(false_positive_rate, true_positive_rate, label=label)
            axis.set_xlabel("False-positive rate")
            axis.set_ylabel("True-positive rate")
            axis.set_title("ROC curve")
        else:
            fraction_positive, mean_predicted_value = calibration_curve(
                y_test,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )
            axis.plot(mean_predicted_value, fraction_positive, marker="o", label=label)
            axis.set_xlabel("Mean predicted probability")
            axis.set_ylabel("Observed bankruptcy rate")
            axis.set_title("Calibration curve")

    if chart_type == "calibration":
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    else:
        axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure


st.title("Corporate Credit Risk")
st.write("Compare bankruptcy predictions from logistic regression and a small neural network.")

with st.sidebar:
    st.header("Data")
    data_path = st.text_input("ARFF file", value="data/5year.arff")
    st.caption("Models train once and are cached until the app is refreshed.")

try:
    X, y, feature_names, comparison, models, holdout = train_dashboard(data_path)
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    st.error(f"Could not load or train the models: {exc}")
    st.stop()

bankruptcy_rate = float(np.mean(y))
summary_columns = st.columns(4)
summary_columns[0].metric("Companies", f"{len(y):,}")
summary_columns[1].metric("Financial ratios", str(X.shape[1]))
summary_columns[2].metric("Bankrupt companies", f"{int(y.sum()):,}")
summary_columns[3].metric("Bankruptcy rate", f"{bankruptcy_rate:.1%}")

st.subheader("Model comparison")
comparison_rows = [
    {
        "Model": "Logistic regression",
        "PR-AUC": comparison.logistic_pr_auc,
        "ROC-AUC": comparison.logistic_auc,
        "Recall": comparison.logistic_recall,
        "Brier score": comparison.logistic_brier,
        "Calibration error": comparison.logistic_ece,
    },
    {
        "Model": "Neural network",
        "PR-AUC": comparison.nn_pr_auc,
        "ROC-AUC": comparison.nn_auc,
        "Recall": comparison.nn_recall,
        "Brier score": comparison.nn_brier,
        "Calibration error": comparison.nn_ece,
    },
]
st.dataframe(comparison_rows, use_container_width=True, hide_index=True)
st.info(f"Preferred model: {comparison.preferred_model.replace('_', ' ').title()}")

chart_columns = st.columns(3)
predictions = {"logistic_regression": holdout["log_probs"], "neural_net": holdout["nn_probs"]}
with chart_columns[0]:
    st.pyplot(metric_chart(holdout["y_test"], predictions, "pr"), use_container_width=True)
with chart_columns[1]:
    st.pyplot(metric_chart(holdout["y_test"], predictions, "roc"), use_container_width=True)
with chart_columns[2]:
    st.pyplot(metric_chart(holdout["y_test"], predictions, "calibration"), use_container_width=True)

st.subheader("Individual company prediction")
company_index = st.number_input(
    "Test-company index",
    min_value=0,
    max_value=len(holdout["X_test"]) - 1,
    value=0,
    step=1,
)
company_row = holdout["X_test"][int(company_index)]
explanation = prediction_explanations(models, company_row)

prediction_columns = st.columns(2)
for column, model_name, title in [
    (prediction_columns[0], "logistic", "Logistic regression"),
    (prediction_columns[1], "neural_net", "Neural network"),
]:
    model_prediction = explanation[model_name]
    with column:
        st.markdown(f"#### {title}")
        st.metric("Bankruptcy probability", f"{model_prediction['probability']:.1%}")
        st.metric("Model risk grade", model_prediction["risk_grade"])
        factors = model_prediction["top_factors"]
        st.dataframe(factors, use_container_width=True, hide_index=True)

with st.expander("Dataset feature names"):
    st.write(", ".join(feature_names))
