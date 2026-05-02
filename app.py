import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Page Config ──────────────────────────────────────────
st.set_page_config(page_title="Social Media Addiction Analyzer", page_icon="📱", layout="centered")

st.title("📱 Social Media Addiction Analyzer")
st.write("Fill in your details to predict addiction level and screen time.")

# ── Load & Prepare Data ───────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("Smartphone_Usage_And_Addiction_Analysis_7500_Rows.csv")
    df.drop(columns=['transaction_id', 'user_id'], inplace=True)
    df['addiction_level'].fillna(df['addiction_level'].mode()[0], inplace=True)

    le_dict = {}
    for col in ['gender', 'stress_level', 'academic_work_impact', 'addiction_level']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Classification
    X_cls = df.drop(columns=['addicted_label', 'daily_screen_time_hours'])
    y_cls = df['addicted_label']
    scaler = StandardScaler()
    X_cls_scaled = scaler.fit_transform(X_cls)
    X_train, X_test, y_train, y_test = train_test_split(X_cls_scaled, y_cls, test_size=0.2, random_state=42)
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    ada.fit(X_train, y_train)

    # Regression
    X_reg = df.drop(columns=['daily_screen_time_hours', 'addicted_label'])
    y_reg = df['daily_screen_time_hours']
    X_train_r, _, y_train_r, _ = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train_r, y_train_r)

    return ada, lr, scaler, le_dict, X_cls.columns.tolist()

ada, lr, scaler, le_dict, feature_cols = load_and_train()

# ── User Input Form ───────────────────────────────────────
st.subheader("📋 Enter Your Details")

age = st.slider("Age", 10, 60, 21)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
social_media_hours = st.slider("Social Media Hours/Day", 0.0, 12.0, 2.0)
gaming_hours = st.slider("Gaming Hours/Day", 0.0, 10.0, 1.0)
work_study_hours = st.slider("Work/Study Hours/Day", 0.0, 12.0, 4.0)
sleep_hours = st.slider("Sleep Hours/Day", 3.0, 12.0, 7.0)
notifications_per_day = st.slider("Notifications Per Day", 0, 500, 100)
app_opens_per_day = st.slider("App Opens Per Day", 0, 300, 80)
weekend_screen_time = st.slider("Weekend Screen Time (hours)", 0.0, 20.0, 5.0)
stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
academic_work_impact = st.selectbox("Academic/Work Impact", ["Yes", "No"])
addiction_level = st.selectbox("Addiction Level (self-assessed)", ["None", "Mild", "Moderate", "Severe"])

# ── Predict Button ────────────────────────────────────────
if st.button("🔍 Predict"):

    gender_enc = le_dict['gender'].transform([gender])[0]
    stress_enc = le_dict['stress_level'].transform([stress_level])[0]
    academic_enc = le_dict['academic_work_impact'].transform([academic_work_impact])[0]

    try:
        addiction_enc = le_dict['addiction_level'].transform([addiction_level])[0]
    except:
        addiction_enc = 0

    input_data = pd.DataFrame([[age, gender_enc, social_media_hours, gaming_hours,
                                  work_study_hours, sleep_hours, notifications_per_day,
                                  app_opens_per_day, weekend_screen_time, stress_enc,
                                  academic_enc, addiction_enc]],
                               columns=feature_cols)

    input_scaled = scaler.transform(input_data)

    # Classification prediction
    addiction_pred = ada.predict(input_scaled)[0]

    # Regression prediction
    screen_time_pred = lr.predict(input_data)[0]

    # ── Results ───────────────────────────────────────────
    st.subheader("📊 Results")

    if addiction_pred == 1:
        st.error("🚨 Prediction: YOU ARE ADDICTED to social media!")
    else:
        st.success("✅ Prediction: You are NOT addicted to social media.")

    st.info(f"📱 Predicted Daily Screen Time: **{round(screen_time_pred, 2)} hours**")

    st.subheader("💡 Recommendations")
    if addiction_pred == 1:
        st.warning("👉 Try limiting social media to 1-2 hours/day")
        st.warning("👉 Take screen breaks every 30 minutes")
        st.warning("👉 Turn off non-essential notifications")
    else:
        st.success("👍 Keep maintaining healthy screen time habits!")