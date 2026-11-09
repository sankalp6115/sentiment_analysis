"""Histogram-based Gradient Boosting Classifier for sentiment classification."""

from sklearn.ensemble import HistGradientBoostingClassifier
from sentiment_analysis.data import prepare_datasets
from sentiment_analysis.embeddings import compute_or_load_embeddings
from sentiment_analysis.evaluate import evaluate_model


def build_hist_gradient_boosting(
    max_iter: int = 1000,
    learning_rate: float = 0.01,
    random_state: int = 42,
) -> HistGradientBoostingClassifier:
    """Build HistGradientBoostingClassifier."""
    return HistGradientBoostingClassifier(
        max_iter=max_iter,
        learning_rate=learning_rate,
        random_state=random_state,
    )


def train_and_evaluate():
    """Load data, train HistGradientBoosting, and evaluate."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets()
    X_train_emb, X_val_emb, X_test_emb = compute_or_load_embeddings(
        X_train, X_val, X_test
    )

    print("Training HistGradientBoostingClassifier...")
    model = build_hist_gradient_boosting()
    model.fit(X_train_emb, y_train)

    val_metrics = evaluate_model(model, X_val_emb, y_val, split_name="Validation")
    test_metrics = evaluate_model(model, X_test_emb, y_test, split_name="Test")
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_and_evaluate()

