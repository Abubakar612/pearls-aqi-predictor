from fastapi import APIRouter
from fastapi.responses import JSONResponse

import boto3
import json
import os


router = APIRouter(tags=["Forecast"])


S3_BUCKET = os.environ.get(
    "S3_BUCKET",
    "pearls-aqi-predictor-071493957773"
)

FORECAST_KEY = os.environ.get(
    "FORECAST_KEY",
    "realtime/latest_forecast.json"
)

s3 = boto3.client("s3")


@router.get("/api/forecast")
async def forecast():

    try:

        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key=FORECAST_KEY
        )

        data = json.loads(
            response["Body"].read().decode("utf-8")
        )

        return JSONResponse(
            content=data
        )

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": "Unable to load forecast",
                "message": str(e)
            }
        )