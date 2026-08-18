import numpy as np

from credit_risk.pipeline import compare_models, grade_from_probability, prediction_explanations


def test_grade_mapping_boundaries():
    assert grade_from_probability(0.00) == "A"
    assert grade_from_probability(0.05) == "A"
    assert grade_from_probability(0.10) == "B"
    assert grade_from_probability(0.25) == "C"
    assert grade_from_probability(0.50) == "D"
    assert grade_from_probability(0.90) == "E"


def test_compare_models_and_explanations_output_shape():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(160, 64))
    signal = (X[:, 0] * 1.2 - X[:, 1] * 0.8 + X[:, 2] * 0.5)
    probs = 1.0 / (1.0 + np.exp(-signal))
    y = (probs > 0.5).astype(int)
    feature_names = [f"ratio_{i+1}" for i in range(X.shape[1])]

    comparison, models, holdout = compare_models(X, y, feature_names, random_state=7)

    assert 0.0 <= comparison.logistic_auc <= 1.0
    assert 0.0 <= comparison.nn_auc <= 1.0
    assert comparison.preferred_model in {"logistic_regression", "neural_net"}

    explanation = prediction_explanations(models, holdout["X_test"][0], top_k=3)
    assert set(explanation.keys()) == {"logistic", "neural_net"}
    assert len(explanation["logistic"]["top_factors"]) == 3
    assert len(explanation["neural_net"]["top_factors"]) == 3
    assert explanation["logistic"]["risk_grade"] in {"A", "B", "C", "D", "E"}
    assert explanation["neural_net"]["risk_grade"] in {"A", "B", "C", "D", "E"}
