from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Pearls AQI Predictor",
    description="AI-powered Air Quality Index forecasting system",
    version="0.1.0",
)

templates = Jinja2Templates(directory="app/templates")

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)


@app.get("/")
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={},
    )


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "application": "Pearls AQI Predictor",
        "version": "0.1.0",
    }