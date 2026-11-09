"""Multi-Layer Perceptron (MLP) Classifier for sentiment classification."""

from sklearn.neural_network import MLPClassifier
from sentiment_analysis.data import prepare_datasets
from sentiment_analysis.embeddings import compute_or_load_embeddings
from sentiment_analysis.evaluate import evaluate_model


def build_mlp(
    hidden_layer_sizes=(256, 64),
    max_iter: int = 50,
    early_stopping: bool = True,
    batch_size: int = 32,
    random_state: int = 42,
) -> MLPClassifier:
    """Build MLPClassifier with early stopping."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        early_stopping=early_stopping,
        batch_size=batch_size,
        random_state=random_state,
    )


def train_and_evaluate():
    """Load data, train MLP, and evaluate."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets()
    X_train_emb, X_val_emb, X_test_emb = compute_or_load_embeddings(
        X_train, X_val, X_test
    )

    print("Training MLPClassifier(hidden_layer_sizes=(256, 64))...")
    model = build_mlp()
    model.fit(X_train_emb, y_train)

    val_metrics = evaluate_model(model, X_val_emb, y_val, split_name="Validation")
    test_metrics = evaluate_model(model, X_test_emb, y_test, split_name="Test")
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_and_evaluate()

