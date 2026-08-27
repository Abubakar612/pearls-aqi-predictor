from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.dashboard import router as dashboard_router
from app.routes.api import router as api_router
from app.routes.forecast import router as forecast_router


app = FastAPI(
    title="Pearls AQI Predictor",
    description="AI-powered Air Quality Index forecasting system",
    version="0.1.0",
)


app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)


app.include_router(dashboard_router)
app.include_router(api_router)
app.include_router(forecast_router)
