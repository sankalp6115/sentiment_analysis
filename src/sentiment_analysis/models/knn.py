"""K-Nearest Neighbors and Nearest Centroid classifiers."""

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import normalize
from sentiment_analysis.data import prepare_datasets
from sentiment_analysis.embeddings import compute_or_load_embeddings
from sentiment_analysis.evaluate import evaluate_model


def build_knn(
    n_neighbors: int = 11,
    metric: str = "cosine",
    n_jobs: int = -1,
) -> KNeighborsClassifier:
    """Build KNeighborsClassifier."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        n_jobs=n_jobs,
    )


def build_nearest_centroid(
    metric: str = "euclidean",
) -> NearestCentroid:
    """Build NearestCentroid classifier."""
    return NearestCentroid(metric=metric)


def train_and_evaluate_knn():
    """Train and evaluate KNeighborsClassifier."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets()
    X_train_emb, X_val_emb, X_test_emb = compute_or_load_embeddings(
        X_train, X_val, X_test
    )

    print("Training KNeighborsClassifier(n_neighbors=11, metric='cosine')...")
    model = build_knn(n_neighbors=11, metric="cosine")
    model.fit(X_train_emb, y_train)

    val_metrics = evaluate_model(model, X_val_emb, y_val, split_name="Validation")
    test_metrics = evaluate_model(model, X_test_emb, y_test, split_name="Test")
    return model, val_metrics, test_metrics


def train_and_evaluate_nearest_centroid():
    """Train and evaluate NearestCentroid on L2-normalized embeddings."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_datasets()
    X_train_emb, X_val_emb, X_test_emb = compute_or_load_embeddings(
        X_train, X_val, X_test
    )

    print("Normalizing embeddings for NearestCentroid...")
    X_train_norm = normalize(X_train_emb)
    X_val_norm = normalize(X_val_emb)
    X_test_norm = normalize(X_test_emb)

    print("Training NearestCentroid(metric='euclidean')...")
    model = build_nearest_centroid(metric="euclidean")
    model.fit(X_train_norm, y_train)

    val_metrics = evaluate_model(model, X_val_norm, y_val, split_name="Validation")
    test_metrics = evaluate_model(model, X_test_norm, y_test, split_name="Test")
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    train_and_evaluate_knn()

