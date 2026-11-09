"""Embedding generation and caching utilities using SentenceTransformers."""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def detect_device() -> str:
    """Detect available accelerator: MPS, CUDA, or fallback to CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
) -> SentenceTransformer:
    """Initialize SentenceTransformer on the selected device."""
    device = device or detect_device()
    model = SentenceTransformer(model_name)
    try:
        model = model.to(device)
    except Exception:
        # Fall back gracefully if device move fails
        model = model.to("cpu")
    return model


def compute_or_load_embeddings(
    X_train,
    X_val,
    X_test,
    cache_dir: str = "dataset",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached numpy embeddings if available, or compute and cache them."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    train_emb_file = cache_path / "X_train_emb.npy"
    val_emb_file = cache_path / "X_val_emb.npy"
    test_emb_file = cache_path / "X_test_emb.npy"

    if (
        not force_recompute
        and train_emb_file.exists()
        and val_emb_file.exists()
        and test_emb_file.exists()
    ):
        print(f"Loading cached embeddings from {cache_path}...")
        X_train_emb = np.load(train_emb_file)
        X_val_emb = np.load(val_emb_file)
        X_test_emb = np.load(test_emb_file)
        return X_train_emb, X_val_emb, X_test_emb

    print(f"Computing embeddings with {model_name}...")
    model = get_embedding_model(model_name=model_name)

    X_train_emb = model.encode(
        X_train.tolist() if hasattr(X_train, "tolist") else list(X_train),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    X_val_emb = model.encode(
        X_val.tolist() if hasattr(X_val, "tolist") else list(X_val),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    X_test_emb = model.encode(
        X_test.tolist() if hasattr(X_test, "tolist") else list(X_test),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(train_emb_file, X_train_emb)
    np.save(val_emb_file, X_val_emb)
    np.save(test_emb_file, X_test_emb)
    print(f"Cached embeddings saved to {cache_path}.")

    return X_train_emb, X_val_emb, X_test_emb

