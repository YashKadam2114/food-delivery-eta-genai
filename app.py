import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="Food Delivery ETA Prediction",
    page_icon="🍔",
    layout="wide"
)

@st.cache_resource
def train_eta_model():
    np.random.seed(42)
    n = 2500

    distance = np.random.uniform(0.5, 15, n)
    prep_time = np.random.normal(18, 6, n).clip(5, 40)
    rider_load = np.random.randint(0, 6, n)
    traffic = np.random.choice(["Low", "Medium", "High"], n, p=[0.30, 0.45, 0.25])
    weather = np.random.choice(["Clear", "Cloudy", "Rain"], n, p=[0.55, 0.25, 0.20])
    hour = np.random.randint(10, 24, n)
    weekend = np.random.choice([0, 1], n, p=[0.7, 0.3])
    restaurant_load = np.random.randint(1, 11, n)

    traffic_effect = pd.Series(traffic).map(
        {"Low": 0, "Medium": 6, "High": 14}
    ).values
    weather_effect = pd.Series(weather).map(
        {"Clear": 0, "Cloudy": 2, "Rain": 8}
    ).values

    peak_effect = np.where(
        ((hour >= 12) & (hour <= 14)) |
        ((hour >= 19) & (hour <= 22)), 5, 0
    )

    actual_delivery = (
        8
        + distance * 2.2
        + prep_time * 0.85
        + rider_load * 2.2
        + traffic_effect
        + weather_effect
        + peak_effect
        + weekend * 2
        + restaurant_load * 0.8
        + np.random.normal(0, 3, n)
    ).clip(10, 120)

    df = pd.DataFrame({
        "distance_km": distance.round(2),
        "restaurant_prep_min": prep_time.round(1),
        "rider_load": rider_load,
        "traffic": traffic,
        "weather": weather,
        "hour": hour,
        "weekend": weekend,
        "restaurant_load": restaurant_load,
        "actual_delivery_min": actual_delivery.round(1)
    })

    X = df.drop(columns=["actual_delivery_min"])
    y = df["actual_delivery_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    categorical = ["traffic", "weather"]
    numerical = [
        "distance_km", "restaurant_prep_min", "rider_load",
        "hour", "weekend", "restaurant_load"
    ]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)

    return model, mae, df

def generate_explanation(order, eta, low, high):
    try:
        from google import genai
        api_key = st.secrets["GEMINI_API_KEY"]
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are the explanation layer of a food delivery ETA prediction system.

Give a concise, customer-friendly explanation of the ETA.
Do not invent any reason. Use ONLY the supplied data.
Mention the ETA and range. Mention the main factors.
Maximum 70 words.

Distance: {order['distance_km']} km
Restaurant preparation: {order['restaurant_prep_min']} minutes
Rider load: {order['rider_load']} orders
Traffic: {order['traffic']}
Weather: {order['weather']}
Time: {order['hour']}:00
Weekend: {"Yes" if order['weekend'] else "No"}
Restaurant load: {order['restaurant_load']}

Predicted ETA: {eta} minutes
Expected range: {low}-{high} minutes
"""

        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return (
            f"ETA is approximately {eta} minutes, with an expected range of "
            f"{low}-{high} minutes. Main factors include {order['traffic'].lower()} "
            f"traffic, {order['weather'].lower()} weather, restaurant preparation "
            f"time and current restaurant/rider workload. "
            f"(GenAI unavailable: {type(e).__name__})"
        )

model, mae, df = train_eta_model()

st.title("🍔 GenAI-Powered Food Delivery ETA Prediction")
st.caption("Machine Learning prediction + GenAI explanation + monitoring")

col1, col2, col3 = st.columns(3)
col1.metric("Model", "Gradient Boosting")
col2.metric("Test MAE", f"{mae:.2f} min")
col3.metric("Training Records", f"{len(df):,}")

st.divider()

st.subheader("📦 New Order")

c1, c2, c3 = st.columns(3)

with c1:
    distance = st.slider("Delivery distance (km)", 0.5, 15.0, 5.2, 0.1)
    prep = st.slider("Restaurant preparation (min)", 5, 40, 20)
    rider = st.slider("Rider current load", 0, 5, 3)

with c2:
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High"], index=2)
    weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain"], index=2)
    hour = st.slider("Order hour", 10, 23, 20)

with c3:
    weekend = st.selectbox("Weekend?", ["No", "Yes"], index=1)
    restaurant_load = st.slider("Restaurant order load", 1, 10, 8)

order = pd.DataFrame([{
    "distance_km": distance,
    "restaurant_prep_min": prep,
    "rider_load": rider,
    "traffic": traffic,
    "weather": weather,
    "hour": hour,
    "weekend": 1 if weekend == "Yes" else 0,
    "restaurant_load": restaurant_load
}])

if st.button("🚀 Predict Delivery ETA", type="primary"):
    eta = float(model.predict(order)[0])
    eta_round = round(eta)
    low = max(10, eta_round - 4)
    high = eta_round + 4

    st.subheader("🤖 Prediction Result")

    r1, r2 = st.columns(2)
    r1.metric("Predicted ETA", f"{eta_round} minutes")
    r2.metric("Expected Range", f"{low}–{high} minutes")

    st.subheader("✨ GenAI Customer Explanation")
    explanation = generate_explanation(order.iloc[0].to_dict(), eta_round, low, high)
    st.info(explanation)

    st.subheader("📊 Model Monitoring")
    actual = st.number_input(
        "Enter actual delivery time after order is completed (minutes)",
        min_value=1, max_value=180, value=eta_round + 3
    )
    error = abs(actual - eta)

    m1, m2 = st.columns(2)
    m1.metric("Actual Delivery", f"{actual} min")
    m2.metric("Absolute Error", f"{error:.2f} min")

    if error > 6:
        st.warning("🚨 Large prediction error detected — investigate possible drift.")
    else:
        st.success("✅ Prediction error is within the monitoring threshold.")

st.divider()

st.subheader("🔄 Production Loop")
st.markdown("""
**Order → Real-time features → ML prediction → ETA → GenAI explanation
→ Actual delivery feedback → Error monitoring → Drift detection → Retraining**
""")

st.caption(
    "Prototype note: the demo uses synthetic data. In production, the model would "
    "be trained on historical delivery records and the monitoring threshold would "
    "be calibrated from business requirements."
)
