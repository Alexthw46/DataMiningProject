import numpy as np


def group_parties(x, parties=['Republican', 'Democrat', 'Other']):
    if x in parties:
        return x
    else:
        return 'Other'


# Converting non-integer values to integers and handle missing values, then check for age range
def clean_age(x):
    try:
        converted = int(x)
    except (ValueError, TypeError):
        return np.nan  # Set a default value for non-convertible or missing values
    if (converted < 0) or (converted > 115):
        return np.nan
    return converted


def to_int(x):
    try:
        converted = int(x)
    except (ValueError, TypeError):
        return np.nan  # Set a default value for non-convertible or missing values
    return converted


def non_negative_int(x):
    try:
        converted = int(x)
    except (ValueError, TypeError):
        return np.nan  # Set a default value for non-convertible or missing values
    if converted < 0:
        return np.nan
    return converted


def sturges_bins(dataframe, column):
    N = dataframe[column].dropna().count()  # Count non-null values in the specified column
    k = int(np.ceil(np.log2(N) + 1))
    return k


# silhouette score
from sklearn.metrics import silhouette_score
# sse / ch index
from sklearn.metrics import calinski_harabasz_score


def calculate_silhouette_scores(data, km_labels, max_clusters=8):
    silhouette_scores = []
    for i in range(2, max_clusters + 1):
        silhouette_scores.append(silhouette_score(data, km_labels[i - 2]))
    return silhouette_scores


def calculate_ch_index(data, km_labels, max_clusters=8):
    ch_index_scores = []
    for i in range(2, max_clusters + 1):
        ch_index_scores.append(calinski_harabasz_score(data, km_labels[i - 2]))
    return ch_index_scores
