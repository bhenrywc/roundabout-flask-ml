# Roundabout Lane Type Predictor

A simple machine learning web application built with Flask. The app predicts whether a roundabout is likely to be **Single-Lane** or **Multilane** based on location and design features.

## Project steps

1. Dataset: Roundabouts Worldwide CSV
2. Model: Random Forest classification model using scikit-learn
3. Web app: Flask form interface
4. Deployment: Render-ready project structure

## Dataset

The dataset contains 27,887 roundabout records with fields such as country, state/region, year completed, number of approaches, driveways, control type, functional class, and lane type.

Original uploaded file: `20_Roundabouts_Worldwide.csv`

## Model Target

The original `lane_type` column has many categories. For a simple beginner-friendly model, this project converts lane type into two classes:

- `Single-Lane`
- `Multilane`

Rows with unknown or rare lane types are removed.

## Run locally

```bash
pip install -r requirements.txt
python src/train_model.py
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Deploy on Render

1. Push this project to GitHub.
2. Go to Render and create a new Web Service.
3. Connect your GitHub repository.
4. Use these settings:
   - Build command: `pip install -r requirements.txt && python src/train_model.py`
   - Start command: `gunicorn app:app`
5. Add your Render link here:

```text
Render URL: paste-your-link-here
```

## Repository description

Flask machine learning web app that predicts whether a roundabout is single-lane or multilane using worldwide roundabout design data.
