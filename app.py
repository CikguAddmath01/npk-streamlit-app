import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
import base64
from io import BytesIO

# ✅ Page config
st.set_page_config(page_title="🌱 Smart NPK Predictor", layout="wide")

# ✅ Load model & scaler
model = load_model('your_lstm_model.h5', compile=False)
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# ✅ Load logo and convert to base64
def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

logo = Image.open('unimap.png')
logo_base64 = image_to_base64(logo)

# ✅ Stage mapping
stage_dict = {'vegetative': 0, 'flowering': 1, 'fruiting': 2}

# ✅ Predict function
def predict_npk(pH, EC, moisture, temperature, humidity, rainfall, stage, week):
    try:
        stage_encoded = stage_dict[stage.lower()]
        features = np.array([[pH, EC, moisture, temperature, humidity, rainfall, stage_encoded, week]])
        X_scaled = scaler_X.transform(features)
        X_seq = X_scaled.reshape((1, 1, X_scaled.shape[1]))
        y_scaled = model.predict(X_seq)
        y_pred = scaler_y.inverse_transform(y_scaled)
        return y_pred[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return [0, 0, 0]

# ✅ Custom CSS Styling
st.markdown(f"""
<style>
body {{
    background-color: #f0fff4;
}}
h1, h2, h3 {{
    color: #22795D;
}}
.big-title {{
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #1B4332;
}}
.card {{
    background-color: #ffffff;
    border-radius: 18px;
    padding: 2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
}}
.metric {{
    font-size: 20px;
}}
img.logo {{
    position: absolute;
    top: 10px;
    right: 20px;
}}
</style>
""", unsafe_allow_html=True)

# ✅ Header
st.markdown(f"""
<img src="data:image/png;base64,{logo_base64}" width="120" class="logo"/>
<div class="big-title">🌾 Smart NPK Prediction Dashboard</div>
<p style='text-align:center; color:#4CAF50'>UniMAP | Powered by LSTM & Smart Farming</p>
""", unsafe_allow_html=True)

# ✅ Input Form
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("predict_form"):
        st.subheader("📥 Input Parameters")
        c1, c2 = st.columns(2)
        with c1:
            pH = st.number_input('Soil pH', 0.0, 14.0, 6.5, 0.1)
            EC = st.number_input('Electrical Conductivity (EC)', 0.0, 10.0, 1.0, 0.1)
            moisture = st.number_input('Moisture (%)', 0.0, 100.0, 30.0, 1.0)
            humidity = st.number_input('Humidity (%)', 0.0, 100.0, 60.0, 1.0)
        with c2:
            temperature = st.number_input('Temperature (°C)', -10.0, 50.0, 28.0, 0.5)
            rainfall = st.number_input('Rainfall (mm)', 0.0, 1000.0, 100.0, 1.0)
            stage = st.selectbox('Growth Stage', ['vegetative', 'flowering', 'fruiting'])
            week = st.slider('Week of Growth', 1, 52, 10)

        submitted = st.form_submit_button("🚀 Predict Now", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ✅ Prediction Output
if submitted:
    result = predict_npk(pH, EC, moisture, temperature, humidity, rainfall, stage, week)
    st.markdown("## 📊 Predicted NPK Values")
    col_n, col_p, col_k = st.columns(3)
    col_n.metric("🌿 Nitrogen (N)", f"{result[0]:.2f} mg/kg")
    col_p.metric("🌼 Phosphorus (P)", f"{result[1]:.2f} mg/kg")
    col_k.metric("🍌 Potassium (K)", f"{result[2]:.2f} mg/kg")
