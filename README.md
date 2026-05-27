# Large-Scale Hybrid Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-Distributed_ETL-orange?logo=apachespark)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy)
![Recommender System](https://img.shields.io/badge/Recommender_System-Hybrid_Model-purple)
![Collaborative Filtering](https://img.shields.io/badge/Collaborative_Filtering-Residual_CF-lightgrey)

A portfolio case study of a large-scale rating prediction system built with Spark, XGBoost, collaborative filtering, and memory-aware feature engineering.

This repository summarizes the system design, modeling strategy, evaluation results, and engineering trade-offs behind a hybrid recommendation engine. It intentionally does not include course-provided data, assignment-specific scripts, or full solution code.

## Overview

This project explores a hybrid recommendation architecture for user-item rating prediction at scale. The system combines metadata-based machine learning, regularized user/item bias modeling, residual collaborative filtering, and lightweight text-derived aggregate features.

The goal was to improve prediction accuracy while keeping the pipeline efficient enough to run under constrained compute and memory environments.

## Highlights

- Built a Spark-based preprocessing pipeline for large-scale user-item interaction data.
- Designed user, item, activity, text, and historical rating feature groups.
- Trained gradient-boosted regression models over multiple feature views.
- Added regularized user/item bias terms for cold-start fallback and baseline prediction.
- Used residual collaborative filtering to capture neighborhood-based signals.
- Improved validation RMSE from approximately `0.9820` to `0.9738`.
- Optimized memory and runtime with `float32` matrices, staged model cleanup, and histogram-based XGBoost training.

## Tech Stack

- Python
- PySpark
- XGBoost
- NumPy
- Collaborative Filtering
- Feature Engineering
- Recommender Systems
- Large-scale ML Pipelines

## System Design

The system combines three complementary prediction sources:

1. **Metadata-based ML model**  
   Learns from user features, item features, activity signals, and historical aggregates.

2. **Regularized bias baseline**  
   Captures user and item rating tendencies while reducing overfitting for sparse users or items.

3. **Residual collaborative filtering**  
   Uses neighborhood-based signals to model deviations from the bias baseline.

The final prediction is produced by blending these components based on confidence and cold-start conditions.

See [`docs/model_architecture.md`](docs/model_architecture.md) for more details.

## Feature Engineering

The feature pipeline includes several groups of signals:

- User metadata and activity features
- Item metadata and activity features
- User/item historical rating statistics
- Smoothed baseline features
- Lightweight text aggregate features
- Category and interaction features
- Cold-start indicators

For training rows, historical aggregates are designed to avoid directly leaking the target label into the feature vector.

See [`docs/feature_engineering.md`](docs/feature_engineering.md) for more details.

## Model Architecture

The final architecture follows a hybrid design:

```text
Raw user/item data
        │
        ▼
Spark-based preprocessing
        │
        ▼
Feature groups
        │
        ├── Metadata features
        ├── Rating history features
        ├── Activity features
        └── Text aggregate features
        │
        ▼
Multi-view XGBoost models
        │
        ├── Bias baseline
        └── Residual collaborative filtering
        │
        ▼
Confidence-aware blending
        │
        ▼
Final rating prediction
```

## Evaluation

The system was evaluated using RMSE for rating prediction.

| Model Version | RMSE |
|---|---:|
| Metadata + XGBoost baseline | ~0.9820 |
| Hybrid model with bias and CF signals | ~0.975 |
| Final multi-view hybrid model | ~0.9738 |

The largest remaining errors were concentrated in extreme ratings, especially unexpected 1-star and 5-star cases. This is common in rating prediction because extreme user experiences often depend on event-specific context that may not be fully captured by historical metadata.

See [`docs/evaluation.md`](docs/evaluation.md) for more details.

## Engineering Challenges

Key engineering challenges included:

- Handling large user/item metadata files efficiently
- Preventing feature leakage from historical aggregates
- Balancing model accuracy with runtime and memory constraints
- Combining model-based and collaborative filtering signals
- Improving cold-start robustness
- Managing large intermediate matrices during model training

## What I Learned

This project helped me understand how to build a practical recommender system that combines machine learning, collaborative filtering, feature engineering, and systems-level optimization.

The most important lesson was that recommendation quality improved not from a single model, but from combining multiple stable signals: user/item bias, metadata features, rating history, text aggregates, and neighborhood-based residuals.

## Repository Scope

This public repository is intended as a portfolio case study. It does not include:

- Full assignment-specific solution code
- Course-provided datasets
- Submission scripts
- Generated prediction files
- Private experimental logs

Instead, it focuses on system design, modeling decisions, evaluation results, and engineering trade-offs.