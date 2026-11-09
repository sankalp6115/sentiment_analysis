"""Random Forest Classifier for sentiment classification."""

from sklearn.ensemble import RandomForestClassifier
from sentiment_analysis.data import prepare_datasets
from sentiment_analysis.embeddings import compute_or_load_embeddings
from sentiment_analysis.evaluate import evaluate_model


def build_random_forest(
    n_estimators: int = 200,
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Build RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def train_and_evaluate():
    """Load data, train Random Forest, and evaluate."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets()
    X_train_emb, X_val_emb, X_test_emb = compute_or_load_embeddings(
        X_train, X_val, X_test
    )

    print("Training RandomForestClassifier(n_estimators=200)...")
    model = build_random_forest(n_estimators=200)
    model.fit(X_train_emb, y_train)

    val_metrics = evaluate_model(model, X_val_emb, y_val, split_name="Validation")
    test_metrics = evaluate_model(model, X_test_emb, y_test, split_name="Test")
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_and_evaluate()

