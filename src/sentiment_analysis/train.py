"""CLI runner to train and benchmark models for sentiment analysis."""

import argparse
from typing import Dict
from sklearn.preprocessing import normalize
from sentiment_analysis.data import prepare_datasets
from sentiment_analysis.embeddings import compute_or_load_embeddings
from sentiment_analysis.evaluate import evaluate_model
from sentiment_analysis.models.logistic_regression import (
    build_logistic_regression,
    build_logistic_regression_cv,
)
from sentiment_analysis.models.random_forest import build_random_forest
from sentiment_analysis.models.svm import build_linear_svc
from sentiment_analysis.models.mlp import build_mlp
from sentiment_analysis.models.gradient_boosting import (
    build_hist_gradient_boosting,
)
from sentiment_analysis.models.knn import (
    build_knn,
    build_nearest_centroid,
)


MODEL_REGISTRY = {
    "logistic": lambda: build_logistic_regression(C=0.5),
    "logistic_cv": lambda: build_logistic_regression_cv(),
    "rf": lambda: build_random_forest(n_estimators=200),
    "svm": lambda: build_linear_svc(C=0.5),
    "mlp": lambda: build_mlp(hidden_layer_sizes=(256, 64), max_iter=50),
    "hgb": lambda: build_hist_gradient_boosting(max_iter=1000, learning_rate=0.01),
    "knn": lambda: build_knn(n_neighbors=11, metric="cosine"),
    "nearest_centroid": lambda: build_nearest_centroid(metric="euclidean"),
}


def run_experiment(model_key: str, embeddings_data, datasets_data) -> Dict:
    """Train and evaluate a specific model."""
    (X_train_emb, X_val_emb, X_test_emb) = embeddings_data
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets_data

    print(f"\n==========================================")
    print(f"Running experiment: {model_key.upper()}")
    print(f"==========================================")

    model_builder = MODEL_REGISTRY[model_key]
    model = model_builder()

    if model_key == "nearest_centroid":
        # Normalize embeddings for cosine/euclidean equivalence
        train_feat = normalize(X_train_emb)
        val_feat = normalize(X_val_emb)
        test_feat = normalize(X_test_emb)
    else:
        train_feat = X_train_emb
        val_feat = X_val_emb
        test_feat = X_test_emb

    model.fit(train_feat, y_train)

    val_res = evaluate_model(model, val_feat, y_val, split_name="Validation")
    test_res = evaluate_model(model, test_feat, y_test, split_name="Test")

    return {
        "model": model_key,
        "val_accuracy": val_res["accuracy"],
        "test_accuracy": test_res["accuracy"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate sentiment analysis models."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Model to run or 'all' to benchmark all models.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="dataset",
        help="Directory where CSVs and .npy embeddings are stored.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputing embeddings instead of using cached .npy files.",
    )
    args = parser.parse_args()

    # 1. Load data
    datasets_data = prepare_datasets(cache_dir=args.cache_dir)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets_data

    # 2. Get embeddings
    embeddings_data = compute_or_load_embeddings(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
    )

    models_to_run = (
        list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    )

    results = []
    for model_key in models_to_run:
        res = run_experiment(model_key, embeddings_data, datasets_data)
        results.append(res)

    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY RESULTS")
    print("=" * 50)
    print(f"{'Model':<20} | {'Val Accuracy':<12} | {'Test Accuracy':<12}")
    print("-" * 50)
    for r in results:
        print(
            f"{r['model']:<20} | {r['val_accuracy']:<12.4f} | {r['test_accuracy']:<12.4f}"
        )
    print("=" * 50)


if __name__ == "__main__":
    main()

