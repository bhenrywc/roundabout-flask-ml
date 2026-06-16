import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "roundabout_lane_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "roundabouts.csv")

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


def get_options():
    df = pd.read_csv(DATA_PATH)
    return {
        "countries": sorted(df["country"].dropna().unique().tolist())[:150],
        "types": sorted(df["type"].dropna().unique().tolist()),
        "statuses": sorted(df["status"].dropna().unique().tolist()),
        "functional_classes": sorted(df["functional_class"].dropna().unique().tolist()),
        "control_types": sorted(df["control_type"].dropna().unique().tolist()),
        "previous_control_types": sorted(df["previous_control_type"].dropna().unique().tolist()),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    options = get_options()

    if request.method == "POST":
        if model is None:
            prediction = "Model not found. Run: python src/train_model.py"
        else:
            input_data = pd.DataFrame(
                [
                    {
                        "country": request.form.get("country"),
                        "state_region": request.form.get("state_region"),
                        "type": request.form.get("type"),
                        "status": request.form.get("status"),
                        "year_completed": int(request.form.get("year_completed")),
                        "approaches": int(request.form.get("approaches")),
                        "driveways": int(request.form.get("driveways")),
                        "functional_class": request.form.get("functional_class"),
                        "control_type": request.form.get("control_type"),
                        "previous_control_type": request.form.get("previous_control_type"),
                    }
                ]
            )

            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            confidence = round(max(probabilities) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        options=options,
    )


if __name__ == "__main__":
    app.run(debug=True)
