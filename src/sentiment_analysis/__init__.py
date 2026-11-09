"""Multiclass sentiment analysis package."""


def main() -> None:
    """CLI entrypoint."""
    from sentiment_analysis.train import main as train_main

    train_main()


__all__ = ["main"]
