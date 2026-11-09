# Multiclass Sentiment Analysis Experiment

A modular machine learning project investigating 3-class sentiment classification on social media data using dense sentence embeddings and classical/shallow ML models.

---

## 📌 1. Project Overview

The objective of this project is to classify short social media texts (tweets) into three distinct sentiment categories:
* **`0` - Negative**
* **`1` - Neutral**
* **`2` - Positive**

The original experiment extracted 384-dimensional dense sentence embeddings using `sentence-transformers/all-MiniLM-L6-v2` in feature-extraction mode, and evaluated multiple classification models on top of these static representations.

---

## 📊 2. Dataset Information

* **Source**: [`Sp1786/multiclass-sentiment-analysis-dataset`](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) (Hugging Face)
* **Dataset Splits**:
  * **Train**: 31,232 samples
  * **Validation**: 5,205 samples
  * **Test**: 5,205 samples
* **Class Distribution**:
  * Neutral: ~37.3% (11,649)
  * Positive: ~33.5% (10,478)
  * Negative: ~29.2% (9,105)
* **Key Text Characteristics**:
  * Informal social media text with typos, slang (`ur`, `idk`, `naw`), hashtags, and user mentions (`@...`).
  * Punctuation-based tone markers (`...`, `???`, `LOL`, `ughhhh`).
  * High proportion of ambiguous neutral tweets.

---

## 🧪 3. Experimental Benchmark Results

Across all models tested on the frozen `all-MiniLM-L6-v2` embeddings, performance clustered in the **60% – 68%** range:

| Model | Hyperparameters / Setup | Validation Accuracy | Test Accuracy | Macro F1 | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **MLP Classifier** | `hidden=(256, 64)`, `early_stopping=True`, `batch=32` | **68.00%** | — | 0.68 | Best performing neural head |
| **Logistic Regression** | `C=0.5`, `max_iter=1000`, L2 | 66.99% | **66.74%** | 0.67 | Strong linear baseline, fast convergence |
| **Logistic Regression CV** | 3-fold CV (`C in [0.01..5.0]`, Best `C=1.0`) | 67.00% | — | 0.67 | CV confirms optimal $C \approx 1.0$ |
| **LinearSVC** | `C=0.5`, `penalty='l2'` | — | 66.80% | 0.67 | Matches Logistic Regression |
| **HistGradientBoosting** | `learning_rate=0.01`, `max_iter=1000` | 66.00% | — | 0.66 | Gradient boosted trees on dense embeddings |
| **Nearest Centroid** | L2 normalized embeddings, Euclidean metric | 62.00% | — | 0.62 | Baseline cosine centroid distance |
| **Random Forest** | `n_estimators=200`, `n_jobs=-1` | — | 61.65% | 0.61 | Decision trees struggle with dense 384-D vectors |
| **K-Nearest Neighbors** | `k=11`, `metric='cosine'` | 60.00% | 60.00% | 0.60 | High computational cost, lower precision |

---

## 🔍 4. Why Accuracy Plateaued at ~68% (Root Cause Analysis)

Every classifier tested stalled between 66% and 68%. This is **not a model selection problem**, but an **information/feature representation bottleneck**:

1. **Semantic Similarity $\neq$ Sentiment Polarity**:
   * `all-MiniLM-L6-v2` was trained with contrastive learning for semantic search and retrieval. In semantic space, *"I love this product"* and *"I hate this product"* have very high cosine similarity because they share identical topical and syntactic structure.
   * Fine-grained sentiment indicators (e.g., negations, sarcasm, exclamation marks) are largely compressed or smoothed out.
2. **Frozen Embedding Bottleneck (Feature Extraction)**:
   * All 8 classifiers were trained on the exact same static 384-dimensional vectors. If the embeddings do not geometrically separate positive, negative, and neutral tweets, no downstream classifier (linear, tree-based, or shallow MLP) can recover the lost information.
3. **The Neutral Class Ambiguity**:
   * Inspection of the classification reports shows neutral class precision (~0.60) and recall (~0.61) are consistently the lowest. Tweets like *"Early monday cramming... yay. Only a few weeks left"* are labeled neutral, creating significant overlap with subtle positive and negative classes.

---

## 🚀 5. Roadmap to Break Past 75%–80%+

To push accuracy well past the 70% threshold:

1. **End-to-End Fine-Tuning with Domain-Specific Transformers (Highest Impact: ~76%–82%)**:
   * Instead of frozen feature extraction, fine-tune a pre-trained transformer model end-to-end with cross-entropy loss.
   * Recommended model: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) (pre-trained on ~124M tweets and fine-tuned for 3-class sentiment: Negative, Neutral, Positive).
2. **Tweet Preprocessing**:
   * Replace `@username` mentions with `@user`.
   * Replace URLs with `http`.
   * Unescape HTML entities (`&amp;` $\rightarrow$ `&`).
3. **Hybrid Feature Engineering (If Staying with Scikit-Learn: ~69%–72%)**:
   * Concatenate character & word n-gram `TfidfVectorizer` (captures exact sentiment keywords, emoticons, and negations) with dense sentence embeddings.
   * Add `class_weight='balanced'` to prevent bias toward the neutral class.

---

## 📂 6. Project Architecture

The original notebook (`sentiment-analysis.ipynb`) has been modularized into production-ready Python files:

```
sentiment-analysis/
├── dataset/                               # Cached CSVs and precomputed .npy embeddings
├── src/
│   └── sentiment_analysis/
│       ├── __init__.py                    # Package entry point
│       ├── data.py                        # Dataset loading, caching, null handling
│       ├── embeddings.py                  # SentenceTransformer encoding & caching (.npy)
│       ├── evaluate.py                    # Accuracy, classification report, confusion matrix
│       ├── train.py                       # Unified CLI runner for benchmarking models
│       └── models/
│           ├── __init__.py                # Model registry exports
│           ├── logistic_regression.py     # LogisticRegression & LogisticRegressionCV
│           ├── random_forest.py           # RandomForestClassifier
│           ├── svm.py                     # LinearSVC
│           ├── mlp.py                     # MLPClassifier (neural net)
│           ├── gradient_boosting.py       # HistGradientBoostingClassifier
│           └── knn.py                     # KNeighborsClassifier & NearestCentroid
├── sentiment-analysis.ipynb               # Original exploration notebook
├── pyproject.toml                         # Project metadata and dependencies
└── README.md                              # Experiment documentation and results
```

---

## 💻 7. How to Run

### Installation & Environment

Using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```
Or with standard Python:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run Models Individually

Each model file can be run directly from the command line:

```bash
# Logistic Regression
python -m sentiment_analysis.models.logistic_regression

# Random Forest
python -m sentiment_analysis.models.random_forest

# Support Vector Machine (LinearSVC)
python -m sentiment_analysis.models.svm

# Multi-Layer Perceptron (Neural Network)
python -m sentiment_analysis.models.mlp

# Histogram Gradient Boosting
python -m sentiment_analysis.models.gradient_boosting

# K-Nearest Neighbors / Nearest Centroid
python -m sentiment_analysis.models.knn
```

### Run Unified Benchmark CLI

You can run individual models or benchmark all of them in a single run:

```bash
# Benchmark all models and output a summary comparison table
python -m sentiment_analysis.train --model all

# Run only a specific model
python -m sentiment_analysis.train --model logistic
python -m sentiment_analysis.train --model mlp
python -m sentiment_analysis.train --model svm
```

> **Note on Performance**: On first run, embeddings are computed and automatically saved to `dataset/*.npy`. Subsequent runs will load the cached embeddings instantly.

