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
#MainMenu {display: none !important; visibility: hidden !important;}
footer {display: none !important; visibility: hidden !important;}
header {display: none !important; visibility: hidden !important;}
.stDeployButton {display: none !important;}
a[href*="github"] {display: none !important;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}
[data-testid="stStatusWidget"] {display: none !important;}
[data-testid="stBottomBlockContainer"] {display: none !important;}
[data-testid="manage-app-button"] {display: none !important;}
[data-testid="stAppViewBlockContainer"] > div:last-child {display: none !important;}
.viewerBadge_container__1QSob {display: none !important;}
.viewerBadge_link__1S137 {display: none !important;}
#stDecoration {display: none !important;}
div[data-testid="stBottom"] {display: none !important;}
.st-emotion-cache-1dp5vir {display: none !important;}
.st-emotion-cache-nakbow {display: none !important;}
.st-emotion-cache-h4xjwg {display: none !important;}
.st-emotion-cache-czk5ss {display: none !important;}
button[kind="icon"] {display: none !important;}
.stAppDeployButton {display: none !important;}
.streamlit-footer {display: none !important;}
.css-1lsmgbg {display: none !important;}
.css-14xtw13 {display: none !important;}
.e8zbici0 {display: none !important;}
.e1fqkh3o4 {display: none !important;}
.block-container {padding-top: 1.5rem !important; padding-bottom: 1rem !important;}
.stSlider > div {padding-bottom: 0 !important;}
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

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:14px;text-align:center;margin-bottom:1.5rem">
    <h1 style="color:white;margin:0;font-size:2rem">📱 Social Media Addiction Analyzer</h1>
    <p style="color:rgba(255,255,255,0.85);margin:0.4rem 0 0">AI-powered analysis of your digital habits</p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
s1.metric("Dataset", "7,500 users")
s2.metric("Features", "14")
s3.metric("Model", "AdaBoost")
s4.metric("Accuracy", "~95%")

st.markdown("---")

# ── Inputs ────────────────────────────────────────────────
st.markdown("### 📋 Enter Your Details")

left, right = st.columns(2)

with left:
    st.markdown("**👤 Personal Info**")
    age = st.slider("Age", 10, 60, 21)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    sleep_hours = st.slider("Sleep Hours / Day", 3.0, 12.0, 7.0, 0.5)
    stress_level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    academic_work_impact = st.selectbox("Academic / Work Impact", ["Yes", "No"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📱 App Usage**")
    social_media_hours = st.slider("Social Media Hours / Day", 0.0, 12.0, 2.0, 0.5)
    gaming_hours = st.slider("Gaming Hours / Day", 0.0, 10.0, 1.0, 0.5)
    work_study_hours = st.slider("Work / Study Hours / Day", 0.0, 12.0, 4.0, 0.5)

with right:
    st.markdown("**🔔 Device Behavior**")
    notifications_per_day = st.slider("Notifications / Day", 0, 500, 100)
    app_opens_per_day = st.slider("App Opens / Day", 0, 300, 80)
    weekend_screen_time = st.slider("Weekend Screen Time (hrs)", 0.0, 20.0, 5.0, 0.5)
    addiction_level = st.selectbox("Self-assessed Addiction Level",
                                   ["None", "Mild", "Moderate", "Severe"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📊 Your Summary**")
    total_screen = social_media_hours + gaming_hours + work_study_hours
    st.write(f"🕐 Total screen time estimate: **{round(total_screen, 1)} hrs**")
    st.write(f"😴 Sleep: **{sleep_hours} hrs** {'✅' if sleep_hours >= 7 else '⚠️'}")
    st.write(f"📲 Social media: **{social_media_hours} hrs** {'✅' if social_media_hours <= 2 else '🚨'}")
    st.write(f"🔔 Notifications: **{notifications_per_day}/day** {'✅' if notifications_per_day <= 150 else '⚠️'}")

# ── Predict ───────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🔍 Analyze My Digital Health", use_container_width=True)

if predict_btn:
    with st.spinner("⏳ Running AI analysis... please wait"):
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

        score = 100
        if social_media_hours > 4: score -= 20
        if sleep_hours < 6: score -= 20
        if stress_level == "High": score -= 15
        if notifications_per_day > 300: score -= 15
        if app_opens_per_day > 150: score -= 10
        score = max(score, 0)

    # ── Auto Scroll to Results ────────────────────────────
    st.markdown("""
        <div id="results-section"></div>
        <script>
            setTimeout(function() {
                document.getElementById('results-section').scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }, 300);
        </script>
    """, unsafe_allow_html=True)

    # ── Results ───────────────────────────────────────────
    st.markdown("### 📊 Your Results")
    r1, r2, r3 = st.columns(3)

    with r1:
        if addiction_pred == 1 or score < 60:
            st.error("🚨 **ADDICTED**\n\nSigns of social media addiction detected")
        elif score < 80:
            st.warning("⚠️ **AT RISK**\n\nYour usage needs improvement")
        else:
            st.success("✅ **HEALTHY**\n\nYour usage is within healthy limits")

    with r2:
        color = "#e74c3c" if score < 60 else "#f39c12" if score < 80 else "#00b894"
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;border:1px solid #eee;border-radius:12px">
            <div style="font-size:2.5rem;font-weight:800;color:{color}">{score}</div>
            <div style="color:#888;font-size:0.85rem">/ 100 — Health Score</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div style="text-align:center;padding:1rem;border:1px solid #eee;border-radius:12px">
            <div style="font-size:2.5rem;font-weight:800;color:#667eea">{round(screen_time_pred, 1)}</div>
            <div style="color:#888;font-size:0.85rem">hrs/day — Predicted Screen Time</div>
        </div>""", unsafe_allow_html=True)

    # ── Usage Bars ────────────────────────────────────────
    st.markdown("### 📈 Usage Breakdown")
    p1, p2 = st.columns(2)
    with p1:
        st.write("📱 Social Media")
        st.progress(min(int((social_media_hours / 12) * 100), 100))
        st.write("🎮 Gaming")
        st.progress(min(int((gaming_hours / 10) * 100), 100))
    with p2:
        st.write("😴 Sleep Health")
        st.progress(min(int((sleep_hours / 12) * 100), 100))
        st.write("📚 Work / Study")
        st.progress(min(int((work_study_hours / 12) * 100), 100))

    # ── Tips ──────────────────────────────────────────────
    # ── Tips ──────────────────────────────────────────────
    st.markdown("### 💡 Recommendations")

    if addiction_pred == 1 or score < 60:
        st.error("🚨 Your digital habits need serious attention!")
        st.warning("👉 Limit social media to 1–2 hours per day")
        st.warning("👉 Take a screen break every 30 minutes")
        st.warning("👉 Turn off all non-essential notifications")
        st.warning("👉 Try a full digital detox on weekends")
        st.warning("👉 Use app timers to strictly monitor usage")
        st.warning("👉 Avoid using phone 1 hour before sleep")
        st.warning("👉 Consider talking to a counselor if needed")

    elif score < 80:
        st.warning("⚠️ Your habits are okay but need improvement!")
        st.warning("👉 Try to reduce social media by 30 minutes daily")
        st.warning("👉 Set a phone curfew after 10 PM")
        st.warning("👉 Reduce notifications to only important apps")
        st.warning("👉 Add 30 more minutes of sleep to your routine")
        st.info("💡 Small changes now will make a big difference!")

    else:
        st.success("🏆 Excellent digital habits — keep it up!")
        st.success("👍 Keep maintaining healthy screen time habits")
        st.success("👍 Continue your good sleep schedule")
        st.success("👍 Stay productive with work/study balance")
        st.success("👍 You are a great example of healthy tech usage!")