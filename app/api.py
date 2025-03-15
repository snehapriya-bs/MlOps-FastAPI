import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any
import uvicorn
from fastapi import FastAPI
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from bikeshare_model import __version__ as model_version
from bikeshare_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()
#app = FastAPI()
#model = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
#bikeshare_rental_pipe= load_pipeline(file_name=model)


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()



example_input = {
    "inputs": [
        {
            "dteday": "2012-11-05",
            "season": "winter",
            "hr": "6am",
            "holiday": "No",
            "weekday": "Mon",
            "workingday":  "Yes",
            "weathersit": "Mist",
            "temp": 6.1,
            "atemp": 3.0014000000000003,
            "hum": 49.0,
            "windspeed": 19.0012,
            "casual": 4,
            "registered": 135,
        }
    ]
}


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Bike Share Model Prection
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    
    results = make_prediction(input_data=input_df.replace({np.nan: None}))

    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results

