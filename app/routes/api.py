from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["API"])


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "application": "Pearls AQI Predictor",
        "version": "0.1.0"
    }
