import numpy as np
import pandas as pd


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
        return pd.NA  # Set a default value for non-convertible or missing values
    if (converted < 0) or (converted > 115):
        return pd.NA
    return converted


def to_int(x):
    try:
        converted = int(x)
    except (ValueError, TypeError):
        return pd.NA  # Set a default value for non-convertible or missing values
    return converted


def non_negative_int(x):
    try:
        converted = int(x)
    except (ValueError, TypeError):
        return pd.NA  # Set a default value for non-convertible or missing values
    if converted < 0:
        return pd.NA
    return converted


def sturges_bins(dataframe, column):
    N = dataframe[column].dropna().count()  # Count non-null values in the specified column
    k = int(np.ceil(np.log2(N) + 1))
    return k
