from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


LOCAL_DATA_PATH = Path(__file__).with_name("SFOC POWER RELATION.csv")
NOTEBOOK_DATA_PATH = Path(
    r"C:\Users\DELL\Desktop\RESOURCES 4\tate_sfoc_power_app\SFOC POWER RELATION.csv"
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = LOCAL_DATA_PATH if LOCAL_DATA_PATH.exists() else NOTEBOOK_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            "Could not find 'SFOC POWER RELATION.csv'. Place it next to app.py."
        )

    data = pd.read_csv(data_path)
    data.columns = [column.strip().upper() for column in data.columns]

    required_columns = {"POWER", "SFOC"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The data file is missing required column(s): {missing}")

    data = data[["POWER", "SFOC"]].dropna()
    data["POWER"] = pd.to_numeric(data["POWER"], errors="coerce")
    data["SFOC"] = pd.to_numeric(data["SFOC"], errors="coerce")
    return data.dropna().sort_values("POWER").reset_index(drop=True)


@st.cache_resource
def train_model(data: pd.DataFrame, degree: int = 3):
    X = data[["POWER"]].to_numpy()
    y = data["SFOC"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    polynomial_features = PolynomialFeatures(degree=degree)
    X_train_poly = polynomial_features.fit_transform(X_train_scaled)
    X_test_poly = polynomial_features.transform(X_test_scaled)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_test = model.predict(X_test_poly)
    metrics = {
        "r2": r2_score(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
    }

    return model, scaler, polynomial_features, metrics


def predict_sfoc(power: float, model, scaler, polynomial_features) -> float:
    power_input = np.array([[power]])
    power_scaled = scaler.transform(power_input)
    power_poly = polynomial_features.transform(power_scaled)
    return float(model.predict(power_poly)[0])


def plot_prediction_curve(data: pd.DataFrame, selected_power: float, selected_sfoc: float):
    power_min = float(data["POWER"].min())
    power_max = float(data["POWER"].max())
    X_grid = np.linspace(power_min, power_max, 400).reshape(-1, 1)

    model, scaler, polynomial_features, _ = train_model(data)
    y_grid = model.predict(polynomial_features.transform(scaler.transform(X_grid)))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(data["POWER"], data["SFOC"], color="#1f77b4", alpha=0.72, label="Actual data")
    ax.plot(X_grid, y_grid, color="#d62728", linewidth=2.5, label="Polynomial prediction")
    ax.scatter(
        [selected_power],
        [selected_sfoc],
        color="#2ca02c",
        edgecolor="black",
        s=110,
        zorder=3,
        label="Selected power",
    )
    ax.set_xlabel("Power")
    ax.set_ylabel("SFOC")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


st.set_page_config(page_title="SFOC Power Predictor", page_icon="P", layout="wide")

st.title("SFOC Prediction from Power")
st.caption("Degree-3 polynomial regression based on the notebook training workflow.")

try:
    data = load_data()
    model, scaler, polynomial_features, metrics = train_model(data)
except Exception as exc:
    st.error(str(exc))
    st.stop()

power_min = float(data["POWER"].min())
power_max = float(data["POWER"].max())
default_power = float(data["POWER"].median())

left, right = st.columns([0.36, 0.64], gap="large")

with left:
    st.subheader("Input")
    power_value = st.number_input(
        "Power",
        min_value=0.0,
        max_value=50000.0,
        value=default_power,
        step=100.0,
        format="%.2f",
    )

    st.slider(
        "Training data power range",
        min_value=power_min,
        max_value=power_max,
        value=(power_min, power_max),
        disabled=True,
    )

    sfoc_prediction = predict_sfoc(power_value, model, scaler, polynomial_features)
    st.metric("Predicted SFOC", f"{sfoc_prediction:.3f}")

    if power_value < power_min or power_value > power_max:
        st.warning(
            "This power is outside the training data range, so the result is an extrapolation."
        )

    st.subheader("Model Check")
    metric_1, metric_2 = st.columns(2)
    metric_1.metric("Test R²", f"{metrics['r2']:.4f}")
    metric_2.metric("Test MAE", f"{metrics['mae']:.3f}")

with right:
    st.subheader("Prediction Curve")
    st.pyplot(plot_prediction_curve(data, power_value, sfoc_prediction), clear_figure=True)

with st.expander("View training data"):
    st.dataframe(data, use_container_width=True, hide_index=True)
