from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["Forecast"])


@router.get("/forecast")
async def forecast_info():
    return {
        "status": "available",
        "source": "AWS prediction API",
        "horizons": [24, 48, 72]
    }
