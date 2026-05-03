import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Social Media Addiction Analyzer",
    page_icon="📱",
    layout="wide"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
a[href*="github"] {display: none;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}
.viewerBadge_container__1QSob {display: none !important;}
.viewerBadge_link__1S137 {display: none !important;}
#stDecoration {display: none !important;}
div[data-testid="stBottom"] {display: none !important;}

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    color: white;
}
.hero p {
    font-size: 1.1rem;
    margin: 0.5rem 0 0;
    opacity: 0.9;
    color: white;
}

/* Metric Cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #e8eaf6;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card .label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #333;
}

/* Section Headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #444;
    border-left: 4px solid #667eea;
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem;
}

/* Result Cards */
.result-addicted {
    background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
}
.result-safe {
    background: linear-gradient(135deg, #55efc4, #00b894);
    color: white;
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
}
.result-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0;
}
.result-subtitle {
    font-size: 0.9rem;
    opacity: 0.9;
    margin: 0.3rem 0 0;
}

/* Score Ring */
.score-container {
    text-align: center;
    padding: 1.5rem;
    background: white;
    border-radius: 12px;
    border: 1px solid #e8eaf6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.score-number {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1;
}
.score-label {
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.3rem;
}

/* Tip Cards */
.tip-card {
    background: #f8f9ff;
    border-left: 3px solid #667eea;
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    color: #444;
}
.tip-card-green {
    background: #f0fff8;
    border-left: 3px solid #00b894;
    padding: 0.7rem 1rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    color: #444;
}

/* Input section */
.input-section {
    background: #fafbff;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e8eaf6;
    margin-bottom: 1rem;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 50px !important;
    border: none !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load & Train ──────────────────────────────────────────
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

    X_cls = df.drop(columns=['addicted_label', 'daily_screen_time_hours'])
    y_cls = df['addicted_label']
    scaler = StandardScaler()
    X_cls_scaled = scaler.fit_transform(X_cls)
    X_train, X_test, y_train, y_test = train_test_split(
        X_cls_scaled, y_cls, test_size=0.2, random_state=42)
    ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    ada.fit(X_train, y_train)

    X_reg = df.drop(columns=['daily_screen_time_hours', 'addicted_label'])
    y_reg = df['daily_screen_time_hours']
    X_train_r, _, y_train_r, _ = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train_r, y_train_r)

    return ada, lr, scaler, le_dict, X_cls.columns.tolist()

ada, lr, scaler, le_dict, feature_cols = load_and_train()

# ── Hero Banner ───────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📱 Social Media Addiction Analyzer</h1>
    <p>AI-powered analysis of your digital habits and screen time</p>
</div>
""", unsafe_allow_html=True)

# ── Dataset Stats ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="label">Dataset Size</div><div class="value">7,500</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="label">Features</div><div class="value">14</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="label">Model</div><div class="value" style="font-size:1.1rem">AdaBoost</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="label">Accuracy</div><div class="value">~95%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Enter Your Details</div>', unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("**👤 Personal Info**")
    age = st.slider("Age", 10, 60, 21)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    sleep_hours = st.slider("Sleep Hours / Day", 3.0, 12.0, 7.0, 0.5)
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    academic_work_impact = st.selectbox("Academic / Work Impact", ["Yes", "No"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("**📱 App Usage**")
    social_media_hours = st.slider("Social Media Hours / Day", 0.0, 12.0, 2.0, 0.5)
    gaming_hours = st.slider("Gaming Hours / Day", 0.0, 10.0, 1.0, 0.5)
    work_study_hours = st.slider("Work / Study Hours / Day", 0.0, 12.0, 4.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("**🔔 Device Behavior**")
    notifications_per_day = st.slider("Notifications / Day", 0, 500, 100)
    app_opens_per_day = st.slider("App Opens / Day", 0, 300, 80)
    weekend_screen_time = st.slider("Weekend Screen Time (hrs)", 0.0, 20.0, 5.0, 0.5)
    addiction_level = st.selectbox("Self-assessed Addiction Level",
                                   ["None", "Mild", "Moderate", "Severe"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Live summary
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("**📊 Your Summary**")
    total_screen = social_media_hours + gaming_hours + work_study_hours
    st.write(f"🕐 Total daily screen time estimate: **{round(total_screen, 1)} hrs**")
    st.write(f"😴 Sleep: **{sleep_hours} hrs** {'✅' if sleep_hours >= 7 else '⚠️'}")
    st.write(f"📲 Social media: **{social_media_hours} hrs** {'✅' if social_media_hours <= 2 else '🚨'}")
    st.write(f"🔔 Notifications: **{notifications_per_day}/day** {'✅' if notifications_per_day <= 150 else '⚠️'}")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Analyze My Digital Health", use_container_width=True)

if predict_btn:
    with st.spinner("Running AI analysis..."):

        gender_enc = le_dict['gender'].transform([gender])[0]
        stress_enc = le_dict['stress_level'].transform([stress_level])[0]
        academic_enc = le_dict['academic_work_impact'].transform([academic_work_impact])[0]
        try:
            addiction_enc = le_dict['addiction_level'].transform([addiction_level])[0]
        except:
            addiction_enc = 0

        input_data = pd.DataFrame([[
            age, gender_enc, social_media_hours, gaming_hours,
            work_study_hours, sleep_hours, notifications_per_day,
            app_opens_per_day, weekend_screen_time, stress_enc,
            academic_enc, addiction_enc
        ]], columns=feature_cols)

        input_scaled = scaler.transform(input_data)
        addiction_pred = ada.predict(input_scaled)[0]
        screen_time_pred = lr.predict(input_data)[0]

        # Health Score
        score = 100
        if social_media_hours > 4: score -= 20
        if sleep_hours < 6: score -= 20
        if stress_level == "High": score -= 15
        if notifications_per_day > 300: score -= 15
        if app_opens_per_day > 150: score -= 10
        score = max(score, 0)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Your Results</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)

    with r1:
        if addiction_pred == 1:
            st.markdown("""
            <div class="result-addicted">
                <p class="result-title">🚨 Addicted</p>
                <p class="result-subtitle">Signs of social media addiction detected</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-safe">
                <p class="result-title">✅ Healthy</p>
                <p class="result-subtitle">Your usage is within healthy limits</p>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="score-container">
            <div class="score-number" style="color: {'#e74c3c' if score < 60 else '#f39c12' if score < 80 else '#00b894'}">
                {score}
            </div>
            <div style="font-size:1rem; color:#888;">/ 100</div>
            <div class="score-label">Digital Health Score</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="score-container">
            <div class="score-number" style="color: #667eea">{round(screen_time_pred, 1)}</div>
            <div style="font-size:1rem; color:#888;">hours/day</div>
            <div class="score-label">Predicted Screen Time</div>
        </div>""", unsafe_allow_html=True)

    # Progress bars
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📈 Usage Breakdown</div>', unsafe_allow_html=True)

    p1, p2 = st.columns(2)
    with p1:
        st.write("📱 Social Media Usage")
        st.progress(min(int((social_media_hours / 12) * 100), 100))
        st.write("🎮 Gaming Usage")
        st.progress(min(int((gaming_hours / 10) * 100), 100))
    with p2:
        st.write("😴 Sleep Health")
        st.progress(min(int((sleep_hours / 12) * 100), 100))
        st.write("📚 Work / Study")
        st.progress(min(int((work_study_hours / 12) * 100), 100))

    # Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">💡 Personalized Recommendations</div>', unsafe_allow_html=True)

    if addiction_pred == 1:
        tips = [
            "Limit social media to 1–2 hours per day",
            "Take a screen break every 30 minutes",
            "Turn off non-essential notifications",
            "Try a digital detox on weekends",
            "Use app timers to monitor usage"
        ]
        for tip in tips:
            st.markdown(f'<div class="tip-card">⚡ {tip}</div>', unsafe_allow_html=True)
    else:
        tips = [
            "Keep maintaining healthy screen time habits",
            "Continue your good sleep schedule",
            "Stay productive with work/study balance",
            "You are doing great — keep it up!"
        ]
        for tip in tips:
            st.markdown(f'<div class="tip-card-green">✅ {tip}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.85rem;'>© 2025 Social Media Addiction Analyzer</p>",
    unsafe_allow_html=True
)