
"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)

import pytest
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer, WeathersitImputer, CustomOutlierHandler


def test_weekday_imputer(sample_input_data):
    imputer = WeekdayImputer()

    X, _ = sample_input_data

    imputer.fit(X)

    transformed = imputer.transform(X)

    assert 'dteday' not in transformed.columns

    assert transformed['weekday'].isnull().sum() == 0

    logging.debug(f'Transformed weekdays {transformed['weekday'].tolist()}')
    expected = {'Sun','Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'}

    assert expected.issubset(set(transformed['weekday'].tolist()))


def test_weathersit_imputer(sample_input_data):

    imputer = WeathersitImputer(column='weathersit')

    X, _ = sample_input_data

    assert 'weathersit' in X.columns

    imputer.fit(X)

    transformed = imputer.transform(X)

    assert transformed['weathersit'].isnull().sum() == 0

    expected_mode = X['weathersit'].mode()[0]

    assert imputer.most_frequent == expected_mode


def test_custom_outlier_handler(sample_input_data):

    X, _ = sample_input_data

    numerical_col = config.model_config_.temp_var

    assert numerical_col in X.columns, f"Column '{numerical_col}' not found in dataset"

    outlier_handler = CustomOutlierHandler(variable=numerical_col)

    outlier_handler.fit(X)

    assert hasattr(outlier_handler, "lower_bound"), "lower_bound attribute not set during fit()"
    assert hasattr(outlier_handler, "upper_bound"), "upper_bound attribute not set during fit()"


    transformed = outlier_handler.transform(X)

    # Check if all values are within the expected range
    assert (transformed[numerical_col] >= outlier_handler.lower_bound).all(), "Lower bound violation detected"
    assert (transformed[numerical_col] <= outlier_handler.upper_bound).all(), "Upper bound violation detected"

    # Ensure at least one outlier was clipped (if applicable)
    if (X[numerical_col] < outlier_handler.lower_bound).sum() > 0 or (X[numerical_col] > outlier_handler.upper_bound).sum() > 0:
        assert not X.equals(transformed), "No outliers were clipped, but expected some changes"





