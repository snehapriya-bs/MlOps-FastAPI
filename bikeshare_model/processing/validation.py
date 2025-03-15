from datetime import datetime
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import pre_pipeline_preparation


def validate_inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    pre_processed = pre_pipeline_preparation(df=input_df)
    validated_data = pre_processed[config.model_config_.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

class DataInputSchema(BaseModel):
    dteday: Optional[datetime]
    season: Optional[str]  # 1: Spring, 2: Summer, 3: Fall, 4: Winter
    hr: Optional[str]  # Hour of the day (0-23)
    holiday: Optional[str]  # 0: No, 1: Yes
    weekday: Optional[str]  # Expecting short format ('Mon', 'Tue', etc.)
    workingday: Optional[str]  # 0: No, 1: Yes
    weathersit: Optional[str]  # 1: Clear, 2: Cloudy, 3: Light Rain, 4: Heavy Rain
    temp: Optional[float]  # Normalized temperature
    atemp: Optional[float]  # Normalized "feels like" temperature
    hum: Optional[float]  # Normalized humidity
    windspeed: Optional[float]  # Normalized wind speed
    yr: Optional[int]  # 0: 2011, 1: 2012
    mnth: Optional[str]  # Month Name

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
