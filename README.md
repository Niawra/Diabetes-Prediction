# Diabetes Prediction — Vercel

Educational Machine Learning demo based on the `Diabetes_Analyst.ipynb` project.

## Files
- `index.html`: browser interface
- `api/predict.py`: Python prediction endpoint (`POST /api/predict`)
- `model/diabetes_model.pkl`: trained Random Forest model and preprocessing artifacts
- `train_model.py`: optional script to retrain locally; place `diabetes_prediction_dataset.csv` in this folder first

## Run locally
1. Use Python 3.12.
2. Install dependencies: `pip install -r requirements.txt`
3. Start a local Vercel development server after installing Vercel CLI: `vercel dev`

## Deploy
Push this folder to a GitHub repository, import that repository in Vercel, and deploy. The project uses Vercel's Python runtime for `api/predict.py`.

## Important
This is an educational demonstration only. It is not a medical diagnostic tool or medical advice.
