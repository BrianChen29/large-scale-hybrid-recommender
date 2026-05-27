"""
Simplified residual collaborative filtering demo.

This file illustrates the idea of predicting residuals over a baseline rather
than predicting raw ratings directly.
"""


def compute_residual_rating(raw_rating, baseline_prediction):
    """Compute the residual between an observed rating and a baseline prediction."""
    return raw_rating - baseline_prediction


def shrink_similarity(raw_similarity, overlap_count, shrinkage=10.0):
    """
    Apply shrinkage to reduce the impact of similarities computed from small overlaps.
    """
    confidence = overlap_count / (overlap_count + shrinkage)
    return raw_similarity * confidence


def residual_cf_prediction(neighbor_residuals, neighbor_similarities):
    """
    Compute a weighted residual prediction from neighboring items.

    Parameters
    ----------
    neighbor_residuals:
        List of residual ratings from similar items.
    neighbor_similarities:
        List of similarity scores corresponding to each neighbor.

    Returns
    -------
    float
        Weighted residual prediction.
    """
    numerator = 0.0
    denominator = 0.0

    for residual, similarity in zip(neighbor_residuals, neighbor_similarities):
        if similarity <= 0:
            continue

        numerator += residual * similarity
        denominator += abs(similarity)

    if denominator == 0:
        return 0.0

    return numerator / denominator