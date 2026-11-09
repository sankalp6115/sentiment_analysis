"""Logistic Regression models for sentiment classification."""

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sentiment_analysis.data import prepare_datasets
from sentiment_analysis.embeddings import compute_or_load_embeddings
from sentiment_analysis.evaluate import evaluate_model


def build_logistic_regression(
    C: float = 0.5,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Build LogisticRegression classifier with L2 regularization."""
    return LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)


def build_logistic_regression_cv(
    Cs=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
    cv: int = 3,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegressionCV:
    """Build cross-validated LogisticRegressionCV classifier."""
    return LogisticRegressionCV(
        Cs=list(Cs),
        cv=cv,
        max_iter=max_iter,
        random_state=random_state,
    )


def train_and_evaluate(use_cv: bool = False):
    """Load data, train Logistic Regression, and evaluate on validation and test sets."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets()
    X_train_emb, X_val_emb, X_test_emb = compute_or_load_embeddings(
        X_train, X_val, X_test
    )

    if use_cv:
        print("Training LogisticRegressionCV...")
        model = build_logistic_regression_cv()
        model.fit(X_train_emb, y_train)
        print(f"Best C parameter: {model.C_[0]}")
    else:
        print("Training LogisticRegression(C=0.5)...")
        model = build_logistic_regression(C=0.5)
        model.fit(X_train_emb, y_train)

    val_metrics = evaluate_model(model, X_val_emb, y_val, split_name="Validation")
    test_metrics = evaluate_model(model, X_test_emb, y_test, split_name="Test")
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_and_evaluate(use_cv=False)

