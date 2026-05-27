# Model Architecture

## Overview

The recommendation engine uses a hybrid architecture that combines model-based prediction, regularized bias modeling, and residual collaborative filtering.

The final prediction is not produced by a single model. Instead, multiple complementary signals are combined to improve robustness across warm, sparse, and cold-start cases.

## Architecture Summary

```text
User-item pair
     │
     ▼
Feature extraction
     │
     ├── Metadata features
     ├── Activity features
     ├── Rating history features
     └── Text aggregate features
     │
     ▼
Model-based prediction
     │
     ├── Gradient-boosted tree models
     │
     ├── Regularized bias baseline
     │
     └── Residual collaborative filtering
     │
     ▼
Confidence-aware blending
     │
     ▼
Final rating prediction
```

## Component 1: Gradient-Boosted Models

The main predictive component uses gradient-boosted decision trees.

This model is effective for structured recommendation features because it can capture nonlinear relationships between:

* User behavior
* Item attributes
* Historical rating statistics
* Activity signals
* Text aggregate features
* Interaction-level features

## Multi-View Modeling

Instead of relying on only one feature set, the system can train models over multiple feature views.

Example views:

| View                | Purpose                                              |
| ------------------- | ---------------------------------------------------- |
| Base metadata view  | Learns from user/item metadata and activity signals  |
| Rating history view | Adds user/item historical rating statistics          |
| Text aggregate view | Adds lightweight sentiment and writing-style signals |
| Topic-style view    | Adds lightweight hashed text/topic signals           |

The goal of multi-view modeling is to create diverse prediction signals that make different types of errors.

## Component 2: Regularized Bias Baseline

The bias baseline estimates rating tendency using:

* Global average rating
* User-level rating bias
* Item-level rating bias

Conceptually:

`prediction = global_average + user_bias + item_bias`

Regularization is important because users or items with very few ratings should not receive overly extreme bias values.

## Why Bias Modeling Helps

The bias model is simple but powerful.

It captures patterns such as:

* Some users consistently rate higher than average
* Some users consistently rate lower than average
* Some items are generally liked or disliked
* Sparse user/item statistics should be smoothed

The bias prediction also provides a useful fallback for cold-start or weak-history cases.

## Component 3: Residual Collaborative Filtering

The collaborative filtering component predicts residuals rather than raw ratings.

Instead of asking:

> What rating will this user give this item?

The residual CF component asks:

> How much should the prediction move away from the bias baseline?

This makes the collaborative filtering signal more stable.

## Residual CF Flow

Observed rating
↓
Subtract bias baseline
↓
Residual rating
↓
Neighbor-based collaborative filtering
↓
Predicted residual adjustment

## Component 4: Confidence-Aware Blending

The final prediction blends several components:

* Gradient-boosted model prediction
* Bias baseline prediction
* Residual collaborative filtering adjustment

The blend should depend on confidence.

For example:

* If user/item history is strong, collaborative filtering can receive more influence.
* If the user or item is sparse, the system should rely more on metadata and bias baseline.
* If neighborhood similarity is weak, the CF signal should be down-weighted.

## Cold-Start Handling

Cold-start cases are handled by falling back to features that do not require dense interaction history.

Useful cold-start signals include:

* User metadata
* Item metadata
* Item category features
* Activity signals
* Smoothed global/user/item averages
* Bias baseline prediction

## Model Trade-Offs

### Benefits

* Combines multiple stable signals
* More robust than a single model
* Handles sparse users/items better
* Captures both metadata and collaborative patterns
* Allows modular experimentation

### Limitations

* More complex than a single-model pipeline
* Requires careful feature leakage prevention
* Needs memory-aware feature construction
* Blending strategy can become difficult to tune
* Extreme ratings remain challenging

## Key Takeaway

The architecture improved performance by combining complementary signals rather than relying on one complex model.

The strongest design pattern was:

`metadata model + regularized bias + residual collaborative filtering + confidence-aware blending`
