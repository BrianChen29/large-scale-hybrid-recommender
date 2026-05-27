# Feature Engineering

## Overview

The feature engineering pipeline combines user metadata, item metadata, activity signals, historical rating behavior, and lightweight text-derived aggregate features.

The goal is to give the model multiple complementary views of each user-item pair while avoiding direct leakage from the target rating.

## Feature Groups

### User Features

User-level features describe historical behavior and activity patterns.

Examples include:

- Average historical rating
- Rating count
- Activity level
- Review count
- Social/activity metadata
- Historical rating variance
- Text-derived aggregate behavior

These features help identify whether a user tends to rate generously, critically, frequently, or sparsely.

### Item Features

Item-level features describe item quality, popularity, metadata, and activity patterns.

Examples include:

- Average historical rating
- Rating count
- Category indicators
- Location-related metadata
- Activity signals
- Review volume
- Text-derived aggregate signals

These features help the model understand item popularity, item type, and general rating tendency.

### Activity Features

Activity features represent behavioral signals beyond raw ratings.

Examples include:

- Check-in volume
- Photo count
- Tip count
- User activity level
- Item activity level

These features can serve as proxies for popularity, engagement, and visibility.

### Rating History Features

Historical rating features are among the strongest signals in rating prediction.

Examples include:

- User historical rating average
- Item historical rating average
- User rating count
- Item rating count
- User rating standard deviation
- Item rating standard deviation
- Smoothed user-item baseline

These features help the model capture long-term rating tendencies.

## Smoothed Baseline Features

A simple average can be unstable for sparse users or items. To reduce overfitting, the system uses smoothed estimates that pull sparse statistics toward the global average.

Conceptually:

```text
smoothed_user_signal = global_average + user_weight * (user_average - global_average)
smoothed_item_signal = global_average + item_weight * (item_average - global_average)
```

The more history a user or item has, the more reliable its historical average becomes.

## Text Aggregate Features

Instead of using a heavy NLP model, the system uses lightweight aggregate text signals.

Examples include:

* Positive word ratio
* Negative word ratio
* Sentiment-like score
* Punctuation ratio
* Review length
* Useful/funny/cool style signals

These features are aggregated at the user and item levels.

This approach is much lighter than training a full text model, but it still gives the system some information about writing style and sentiment patterns.

## Topic-Like Features

The system can also use lightweight hashed topic-style features.

The idea is to tokenize review text, remove common words, and map remaining terms into a fixed-size representation.

This provides a low-cost approximation of semantic diversity without requiring large TF-IDF or embedding matrices.

## Interaction Features

Interaction features describe the relationship between a user and an item.

Examples include:

* Difference between user average rating and item average rating
* User activity × item activity
* Cold-start indicators
* Smoothed baseline estimate
* User/item history availability

These features help the model reason about whether a specific user-item pair is likely to behave differently from the global trend.

## Leakage Prevention

A key challenge in rating prediction is avoiding feature leakage.

For training examples, historical aggregates should not directly include the target row's own rating when that would reveal the label.

Examples of leakage-aware design:

* Use leave-one-out style statistics for training rows
* Avoid using validation/test labels during feature construction
* Build rating history features only from training interactions
* Keep text aggregates separate from target labels

## Why Feature Engineering Matters

The final model performance improved through the combination of many stable signals rather than one single complex model.

The most useful feature categories were:

* Historical rating statistics
* User/item metadata
* Activity signals
* Bias baseline features
* Lightweight text aggregates
* Cold-start indicators

Together, these features gave the model a stronger and more robust representation of user-item rating behavior.
