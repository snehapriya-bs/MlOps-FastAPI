"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
from sklearn.metrics import mean_absolute_error

from bikeshare_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = len(sample_input_data[0])

    # When
    result = make_prediction(input_data=sample_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
   
   

    predictions = np.round(predictions).astype(int)  # Convert if needed
    assert isinstance(predictions[0].item(), int), f"Expected Python int but got {type(predictions[0])}"



    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions


    # _predictions = list(predictions)
    y_true = sample_input_data[1]
    mae = mean_absolute_error(y_true, predictions)
    assert mae > 0.8

