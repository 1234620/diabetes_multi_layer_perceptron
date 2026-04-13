Diabetes MLP UI

Overview

Doctor-facing UI to upload a trained Keras model and its meta, then enter patient data to predict diabetes risk.

Structure

- backend/: FastAPI server exposing /upload and /predict, serves frontend
- frontend/: Static HTML/CSS/JS doctor console

Requirements

- Python 3.10+
- Windows PowerShell or any shell

Setup & Run (Backend)

1) Create venv (recommended)

PowerShell

python -m venv .venv
.venv\Scripts\Activate.ps1

2) Install dependencies

pip install -r backend/requirements.txt

3) Start server

python backend/main.py

This serves the frontend at http://localhost:8020/ and the API under the same origin.

Usage

1) Open http://localhost:8020/
2) Upload diabetes_mlp.keras and diabetes_meta.json
3) Enter patient data and click Predict Diabetes

Notes on Meta File

For best results, the meta JSON should include:
- feature_columns: array of feature names used during training (after get_dummies)
- classes_: array of class names (for binary, two labels)
- scaler_mean_ and scaler_scale_: arrays from training StandardScaler

If scaler values are missing, the server uses identity scaling. If classes_ is missing, labels default to Positive/Negative (binary) or index (multiclass).

Docker

Build image:

docker build -t mlp-ui:latest .

Run container:

docker run --rm -p 8020:8020 mlp-ui:latest

Open:

http://localhost:8020/

Jenkins

This repo includes a root-level Jenkins declarative pipeline in `Jenkinsfile`.

What it does:
- Checks out source
- Builds Docker image (`mlp-ui`)
- Recreates and runs container (`mlp-ui-app`) on port `8020`
- Verifies app health at `/`

Jenkins prerequisites:
- Jenkins agent has Docker CLI and daemon access
- Jenkins agent has `curl` installed

Create a Pipeline job and point it to this repository. Jenkins will auto-detect and run `Jenkinsfile`.


