# Evaluation

## Metric

The system was evaluated using Root Mean Squared Error, or RMSE.

RMSE is commonly used for rating prediction because it penalizes larger prediction errors more heavily.

```text
RMSE = sqrt(mean((prediction - actual_rating)^2))
```

Lower RMSE indicates better prediction accuracy.

## Result Summary

| Model Version                         |    RMSE |
| ------------------------------------- | ------: |
| Metadata + XGBoost baseline           | ~0.9820 |
| Hybrid model with bias and CF signals |  ~0.975 |
| Final multi-view hybrid model         | ~0.9738 |

The final system improved over the baseline by combining metadata features, rating history features, regularized bias modeling, residual collaborative filtering, and lightweight text-derived aggregate features.

## Interpretation

The improvement from the baseline was not caused by a single model change.

Instead, performance improved through multiple incremental gains:

* Better user/item metadata representation
* Historical rating statistics
* Smoothed bias baseline
* Residual collaborative filtering
* Text aggregate features
* Multi-view modeling
* Runtime and memory optimization

## Error Pattern

The model generally performs better on common mid-range and high-frequency rating patterns.

The largest remaining errors tend to come from extreme ratings, especially unexpected 1-star and 5-star cases.

This is common in rating prediction because extreme reviews often depend on event-specific context, such as:

* A one-time bad experience
* Unusually good service
* Personal preference mismatch
* Context not visible from metadata
* Sparse user/item history

## Why Extreme Ratings Are Difficult

A user may usually rate positively but suddenly give a 1-star rating because of a specific bad event.

Similarly, a user may give a 5-star rating because of a personal or situational reason that is not captured by historical metadata.

These cases are difficult for structured models because the necessary context may not exist in the available features.

## Engineering Evaluation

Besides model accuracy, the system was also evaluated from an engineering perspective.

Important engineering goals included:

* Keeping preprocessing scalable
* Avoiding excessive memory usage
* Reducing unnecessary intermediate artifacts
* Training models in stages
* Keeping the pipeline stable under constrained runtime
* Avoiding feature leakage

## Lessons Learned

The most important lesson was that recommendation performance improved through the accumulation of many stable signals.

The strongest signals were:

* User/item historical rating behavior
* Smoothed bias estimates
* Metadata and activity features
* Lightweight text aggregates
* Collaborative residual signals

The project also showed that practical ML systems require both modeling improvements and engineering optimizations.

## Future Improvements

Potential future improvements include:

* More robust cold-start modeling
* Better extreme-rating detection
* Calibrated prediction intervals
* More advanced text representation
* Online feature updates
* Experiment tracking with MLflow or Weights & Biases
* Production-style batch inference pipeline
