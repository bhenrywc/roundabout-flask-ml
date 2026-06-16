import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "roundabouts.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "roundabout_lane_model.pkl")


def simplify_lane_type(value):
    value = str(value).lower()
    if "single" in value:
        return "Single-Lane"
    if "multi" in value or "ring junction" in value:
        return "Multilane"
    return None


def main():
    df = pd.read_csv(DATA_PATH)

    df["target_lane_type"] = df["lane_type"].apply(simplify_lane_type)
    df = df.dropna(subset=["target_lane_type"])

    features = [
        "country",
        "state_region",
        "type",
        "status",
        "year_completed",
        "approaches",
        "driveways",
        "functional_class",
        "control_type",
        "previous_control_type",
    ]

    X = df[features]
    y = df["target_lane_type"]

    categorical_features = [
        "country",
        "state_region",
        "type",
        "status",
        "functional_class",
        "control_type",
        "previous_control_type",
    ]
    numeric_features = ["year_completed", "approaches", "driveways"]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=60,
                    max_depth=14,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
