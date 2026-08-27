Yes — **the GitHub link is there because a README can include a repository section, but you don't need it**. Since the README is already inside that GitHub repository, adding the same repository link is unnecessary.

Also, you asked to **paste the complete README exactly as one README**, so remove the `## Repository` section entirely.

Use the version below. It starts with `# PEARLS AQI Predictor` and ends with the author, with **no GitHub repository link** and no architecture diagram.

````markdown
# PEARLS AQI Predictor

An end-to-end machine-learning system for real-time Air Quality Index (AQI) monitoring and 24-hour, 48-hour and 72-hour AQI forecasting for Lahore, Pakistan.

The system combines realtime environmental data, feature engineering, machine-learning models, explainable AI, AWS cloud services, automated pipelines and a web dashboard to provide an operational AQI prediction platform.

---

## Overview

PEARLS AQI Predictor is designed to provide both current air-quality information and short-term AQI forecasts.

The system:

- Collects realtime air-quality and environmental data
- Calculates AQI from pollutant concentrations
- Generates production-ready machine-learning features
- Predicts AQI for the next 24, 48 and 72 hours
- Provides SHAP-based model explainability
- Stores realtime features, forecasts and production models in Amazon S3
- Runs prediction services using AWS Lambda
- Exposes predictions through Amazon API Gateway
- Displays results through a web dashboard
- Sends Amazon SNS alerts for hazardous AQI conditions
- Uses GitHub Actions for automated testing and model training
- Supports scheduled operational processing

---

## Key Features

### Current AQI Monitoring

The system calculates the current AQI using pollutant measurements including:

- PM2.5
- PM10
- NO2
- O3
- CO

The AQI is calculated from pollutant sub-indices and the highest applicable sub-index determines the overall AQI.

---

### Machine-Learning Forecasting

Three production forecasting models are used:

| Forecast Horizon | Output |
|---|---|
| 24 hours | Predicted AQI |
| 48 hours | Predicted AQI |
| 72 hours | Predicted AQI |

The models use engineered temporal, lag, rolling-statistical, weather and pollutant features.

The production feature set contains **25 features**.

---

### Explainable AI

SHAP (SHapley Additive exPlanations) is used to analyse model behaviour and identify the features that have the greatest influence on predictions.

SHAP analysis has been completed for:

- 24-hour forecasting
- 48-hour forecasting
- 72-hour forecasting

Generated outputs include feature-importance CSV files and SHAP summary plots.

Results are stored under:

```text
data/processed/model_analysis/shap/
````

---

## AWS Cloud Infrastructure

The production system is deployed using Amazon Web Services (AWS).

The main AWS services used are:

* Amazon S3
* AWS Lambda
* Amazon API Gateway
* Amazon SNS
* Amazon CloudFront
* Amazon ECR
* AWS IAM

Amazon S3 is used to store realtime data, engineered features, production models and the latest forecast results.

AWS Lambda provides the serverless feature-generation and prediction services.

Amazon API Gateway exposes the prediction service through an HTTP endpoint.

Amazon SNS is used to send alerts when the current AQI or forecast reaches the hazardous AQI threshold.

Amazon CloudFront provides the public web delivery layer for the dashboard.

Amazon ECR stores the Docker container images used for Lambda deployment.

AWS IAM roles and policies control access between deployed AWS services.

---

## Realtime Data Processing

The realtime processing pipeline receives the latest environmental observations for Lahore and stores them in Amazon S3.

The feature-generation Lambda:

1. Reads the latest realtime dataset.
2. Calculates pollutant-specific AQI values.
3. Calculates the overall AQI.
4. Generates the required production features.
5. Validates the production feature list.
6. Selects the latest complete observation.
7. Stores the resulting feature record in Amazon S3.

The production feature file is stored at:

```text
realtime/latest_features.csv
```

The production feature set contains 25 features.

---

## Feature Engineering

The forecasting models use a combination of temporal, historical, pollutant and weather-related features.

Examples include:

* AQI lag features
* PM2.5 lag features
* PM10 lag features
* AQI rolling means
* AQI rolling standard deviations
* PM2.5 rolling statistics
* PM10 rolling statistics
* AQI change features
* PM2.5 change features
* PM10 change features
* Hour
* Day
* Day of week
* Weekend information
* Cyclical time features
* Feels-like temperature

Lag and rolling features are calculated using previous observations to support time-series forecasting.

---

## Production Machine-Learning Models

Three production models are maintained for different forecasting horizons:

```text
models/production/aqi_model_24h.joblib
models/production/aqi_model_48h.joblib
models/production/aqi_model_72h.joblib
```

The production feature definition is stored in:

```text
models/production/feature_list.json
```

The prediction Lambda loads these models from Amazon S3 when generating forecasts.

---

## Prediction API

The machine-learning prediction service is exposed through Amazon API Gateway.

The production route is:

```text
POST /predict
```

A prediction request can be sent using an empty JSON object:

```json
{}
```

The API returns:

* City
* Country
* Current AQI
* Data timestamp
* Pollutant concentrations
* 24-hour forecast
* 48-hour forecast
* 72-hour forecast
* AQI category for each forecast

The latest forecast is also stored in:

```text
realtime/latest_forecast.json
```

---

## Hazardous AQI Alerting

The system includes automated hazardous-air-quality alerts using Amazon SNS.

The configured hazardous threshold is:

```text
AQI >= 301
```

The prediction Lambda checks:

* Current AQI
* 24-hour forecast
* 48-hour forecast
* 72-hour forecast

If any value reaches the hazardous threshold, an SNS notification is published.

The alert contains:

* Location
* Data timestamp
* Hazardous AQI readings or forecasts
* AQI threshold
* Precautionary message

A dedicated test mode was implemented to verify the SNS alert mechanism without waiting for a real hazardous AQI condition.

The alert test successfully returned:

```text
alert_sent: true
```

---

## Automated Pipelines

GitHub Actions is used to automate project validation and model-training activities.

### Feature Pipeline

The feature workflow is located at:

```text
.github/workflows/feature_pipeline.yml
```

The workflow:

* Runs on relevant repository changes
* Runs on pull requests
* Sets up Python 3.11
* Installs project dependencies
* Runs data tests
* Runs feature-engineering tests

---

### Training Pipeline

The training workflow is located at:

```text
.github/workflows/training_pipeline.yml
```

The training workflow runs automatically every day at **02:00 UTC**.

It also supports:

* Push-based execution
* Pull requests
* Manual execution

The workflow:

* Configures AWS credentials through GitHub Secrets
* Runs the complete test suite
* Executes the training pipeline
* Uploads production models to Amazon S3
* Uploads the production feature list
* Verifies the production model files

The deployed operational system also supports scheduled realtime processing.

---

## Model Explainability

SHAP is used to provide model explainability for the production forecasting models.

The analysis was performed using 500 samples for each forecasting horizon.

The generated files include:

```text
shap_importance_24h.csv
shap_importance_48h.csv
shap_importance_72h.csv

shap_summary_24h.png
shap_summary_48h.png
shap_summary_72h.png
```

The analysis identifies the features that contribute most strongly to the predictions.

For the 24-hour model, important features include O3, target AQI, day, cyclical day-of-week information, feels-like temperature and recent AQI values.

For the 48-hour model, important features include O3, day, day of week, feels-like temperature, cyclical day-of-week information and target AQI.

For the 72-hour model, important features include day, O3, day of week, cyclical day-of-week information, target AQI and CO.

---

## AQI Calculation Validation

During production testing, an issue was identified where the system returned a current AQI of 17 despite PM2.5 and PM10 concentrations that should have produced a substantially higher AQI.

The issue was traced to gaps between pollutant breakpoint ranges.

The PM2.5 breakpoint was updated from:

```text
35.4
```

to:

```text
35.49
```

The PM10 breakpoint was updated from:

```text
154
```

to:

```text
154.99
```

This prevented valid pollutant values from falling between breakpoint intervals and producing `NaN`.

After deployment, the feature Lambda correctly calculated:

```text
current_aqi: 150.0
```

The corrected value was then successfully propagated through the prediction Lambda and API.

---

## Production Validation

The corrected production pipeline was tested using the latest realtime observation.

The validated observation was:

```text
Timestamp: 2026-08-27 18:00:00+00:00
Location: Lahore, Pakistan
```

### Current AQI

```text
150.0
```

### Pollutants

| Pollutant |  Value |
| --------- | -----: |
| PM2.5     |  55.44 |
| PM10      | 154.50 |
| NO2       |  16.03 |
| O3        |  36.16 |
| CO        | 450.16 |

### Forecast Results

| Forecast |    AQI | Category                       |
| -------- | -----: | ------------------------------ |
| 24 hours | 161.90 | Unhealthy                      |
| 48 hours | 145.77 | Unhealthy for Sensitive Groups |
| 72 hours | 141.58 | Unhealthy for Sensitive Groups |

The current AQI is dynamically calculated from the latest realtime data and is therefore not permanently fixed at 150.

---

## Web Dashboard

The project includes a web dashboard for displaying the latest AQI information and machine-learning forecasts.

The dashboard provides:

* Current AQI
* AQI category
* Location
* Last updated timestamp
* Pollutant measurements
* 24-hour forecast
* 48-hour forecast
* 72-hour forecast
* Forecast chart
* AQI category indicators

The dashboard communicates with the production prediction API and updates its displayed values using the returned JSON response.

The dashboard is deployed using Amazon S3 and Amazon CloudFront.

---

## Project Structure

```text
pearls-aqi-predictor/
│
├── app/
├── config/
├── data/
├── deploy/
├── lambda/
├── models/
├── notebooks/
├── src/
├── tests/
├── .github/
│   └── workflows/
│
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── README.md
└── shap_analysis.py
```

---

## Testing

The project contains automated tests covering the main data, feature and machine-learning components.

Run the complete test suite using:

```bash
pytest tests/ -v
```

The project also performs production validation of:

* Feature generation
* S3 storage
* Production model loading
* Forecast generation
* API responses
* Hazardous AQI alerts

---

## Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/abubakar612/pearls-aqi-predictor.git
```

### 2. Move into the project directory

```bash
cd pearls-aqi-predictor
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the tests

```bash
pytest tests/ -v
```

---

## Running SHAP Analysis

The SHAP analysis can be executed using:

```bash
python shap_analysis.py
```

The analysis produces outputs under:

```text
data/processed/model_analysis/shap/
```

---

## API Testing

The production API accepts:

```text
POST /predict
Content-Type: application/json
```

Example request:

```json
{}
```

The API returns the latest current AQI, pollutant values and forecasts for 24, 48 and 72 hours.

---

## Technologies Used

### Programming

* Python
* JavaScript
* HTML
* CSS

### Machine Learning

* pandas
* NumPy
* scikit-learn
* SHAP
* joblib

### Cloud

* Amazon S3
* AWS Lambda
* Amazon API Gateway
* Amazon SNS
* Amazon CloudFront
* Amazon ECR
* AWS IAM

### Development and Automation

* Git
* GitHub
* GitHub Actions
* Docker
* Jupyter
* pytest

---

## Security

AWS access is controlled using IAM policies and roles.

GitHub Actions uses encrypted GitHub Secrets for AWS authentication rather than storing AWS credentials directly in workflow files.

Sensitive configuration should be stored using environment variables or protected configuration files.

The `.env` file must never be committed to the repository.

---

## Limitations

The current implementation focuses on Lahore, Pakistan.

Prediction performance can be affected by:

* Data quality
* Missing observations
* Weather changes
* Changes in pollution patterns
* Limited historical data
* Changes in environmental conditions

The forecasting models provide estimates and should not be treated as guaranteed future AQI values.

The current AQI represents the latest available observation and is recalculated when new realtime data is processed.

---

## Future Improvements

Possible future improvements include:

* Support for additional Pakistani cities
* Larger historical datasets
* Automated model-drift detection
* Additional environmental variables
* More advanced forecasting architectures
* Improved alert personalisation
* User-specific notifications
* Long-term model monitoring
* Automated retraining based on model performance

---

## Project Status

**Production system: Operational**

Completed components:

* [x] Realtime data processing
* [x] AQI calculation
* [x] Feature engineering
* [x] Production feature generation
* [x] 24-hour forecasting
* [x] 48-hour forecasting
* [x] 72-hour forecasting
* [x] SHAP explainability
* [x] AWS S3 integration
* [x] AWS Lambda deployment
* [x] API Gateway integration
* [x] Amazon SNS hazardous AQI alerting
* [x] Web dashboard
* [x] CloudFront deployment
* [x] GitHub Actions automation
* [x] Automated testing
* [x] Production validation
* [x] AQI breakpoint correction

---

## Recent GitHub Commits

Recent project improvements include:

```text
47808d7 Fix AQI breakpoint boundary handling
017ab99 Add AQI alerts and production feature generation
50b9588 Add SHAP explainability analysis
7b6357c Fix test discovery in training workflow
39c7362 Add boto3 dependency
```

The repository has been verified with a clean working tree and is synchronised with the main branch.

---

## Author

**Abubakar Farooq**

PEARLS AQI Predictor

Machine-Learning-Based Air Quality Monitoring and Forecasting System

```

**One small note:** the `git clone` command contains the GitHub URL because that is genuinely useful for someone installing the project. The separate **Repository** section/link has been removed.
```
