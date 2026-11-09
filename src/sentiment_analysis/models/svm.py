"""Support Vector Classifier for sentiment classification."""

from sklearn.svm import LinearSVC
from sentiment_analysis.data import prepare_datasets
from sentiment_analysis.embeddings import compute_or_load_embeddings
from sentiment_analysis.evaluate import evaluate_model


def build_linear_svc(
    C: float = 0.5,
    penalty: str = "l2",
    dual: str = "auto",
    max_iter: int = 2000,
    random_state: int = 42,
) -> LinearSVC:
    """Build LinearSVC classifier."""
    return LinearSVC(
        C=C,
        penalty=penalty,
        dual=dual,
        max_iter=max_iter,
        random_state=random_state,
    )


def train_and_evaluate():
    """Load data, train LinearSVC, and evaluate."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets()
    X_train_emb, X_val_emb, X_test_emb = compute_or_load_embeddings(
        X_train, X_val, X_test
    )

    print("Training LinearSVC(C=0.5)...")
    model = build_linear_svc(C=0.5)
    model.fit(X_train_emb, y_train)

    val_metrics = evaluate_model(model, X_val_emb, y_val, split_name="Validation")
    test_metrics = evaluate_model(model, X_test_emb, y_test, split_name="Test")
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_and_evaluate()

