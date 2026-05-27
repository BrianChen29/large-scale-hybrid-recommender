"""
Example feature schema for a hybrid recommendation system.

This file lists high-level feature groups without exposing dataset-specific
feature extraction logic.
"""


USER_FEATURES = [
    "user_avg_rating",
    "user_rating_count",
    "user_activity_level",
    "user_text_sentiment_aggregate",
    "user_rating_std",
]


ITEM_FEATURES = [
    "item_avg_rating",
    "item_rating_count",
    "item_activity_level",
    "item_category_indicators",
    "item_text_sentiment_aggregate",
]


INTERACTION_FEATURES = [
    "user_item_avg_diff",
    "user_item_activity_interaction",
    "smoothed_bias_baseline",
    "cold_start_indicator",
]


TEXT_AGGREGATE_FEATURES = [
    "positive_word_ratio",
    "negative_word_ratio",
    "punctuation_ratio",
    "text_length_log",
]


RATING_HISTORY_FEATURES = [
    "user_history_count",
    "item_history_count",
    "user_history_std",
    "item_history_std",
    "smoothed_user_item_baseline",
]