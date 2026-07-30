from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "optimized_sleep_apnea_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "models" / "optimized_model_features.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if FEATURES_PATH.exists():
        features = joblib.load(FEATURES_PATH)
    else:
        features = ["HR_Mean", "SpO2_Mean", "Flow_Mean"]

    return model, features


st.set_page_config(
    page_title="Sleep Apnea Detection",
    page_icon="🌙",
    layout="centered"
)

st.title("🌙 Sleep Apnea Detection")
st.write(
    "Enter physiological measurements to receive a prediction "
    "from the optimized Random Forest model."
)

try:
    model, feature_names = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


heart_rate = st.number_input(
    "Average Heart Rate",
    min_value=20.0,
    max_value=220.0,
    value=75.0,
    step=1.0
)

spo2 = st.number_input(
    "Average SpO₂ (%)",
    min_value=50.0,
    max_value=100.0,
    value=95.0,
    step=0.1
)

airflow = st.number_input(
    "Average Airflow",
    value=0.0,
    step=0.01,
    format="%.4f"
)


if st.button("Predict"):
    input_data = pd.DataFrame(
        [[heart_rate, spo2, airflow]],
        columns=["HR_Mean", "SpO2_Mean", "Flow_Mean"]
    )

    input_data = input_data[feature_names]

    prediction = int(model.predict(input_data)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        class_probabilities = dict(zip(model.classes_, probabilities))

        non_apnea_probability = class_probabilities.get(0, 0.0)
        apnea_probability = class_probabilities.get(1, 0.0)

        st.write(
            f"Apnea probability: **{apnea_probability:.2%}**"
        )
        st.write(
            f"Non-apnea probability: **{non_apnea_probability:.2%}**"
        )

    if prediction == 1:
        st.warning("Model prediction: Apnea")
    else:
        st.success("Model prediction: Non-Apnea")

    st.caption(
        "This project is an educational machine-learning demonstration "
        "and is not a medical diagnosis."
    )