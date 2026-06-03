"""
Diabetes Risk Prediction System
Streamlit Community Cloud deployment
Uses pre-trained XGBoost model + StandardScaler to ensure 100% consistency with notebook.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# Custom CSS — clean clinical white + teal (unchanged)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F0F4F8;
    color: #1A2332;
}

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0A4F6B 0%, #0E7490 60%, #06B6D4 100%);
    border-radius: 16px;
    padding: 36px 44px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 8px 32px rgba(10,79,107,0.18);
}
.header-icon { font-size: 52px; }
.header-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #ffffff;
    margin: 0;
    line-height: 1.15;
}
.header-sub {
    font-size: 0.92rem;
    color: rgba(255,255,255,0.78);
    margin-top: 4px;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Section headers ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #0E7490;
    margin-bottom: 10px;
    margin-top: 4px;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem;
    color: #1A2332;
    margin-bottom: 18px;
    border-left: 3px solid #0E7490;
    padding-left: 12px;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border-radius: 14px;
    padding: 24px 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #E2EBF0;
    margin-bottom: 18px;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 10px 28px;
    border-radius: 40px;
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
}
.risk-low    { background:#D1FAE5; color:#065F46; border:1.5px solid #34D399; }
.risk-medium { background:#FEF3C7; color:#92400E; border:1.5px solid #FBBF24; }
.risk-high   { background:#FEE2E2; color:#991B1B; border:1.5px solid #F87171; }

/* ── Probability number ── */
.prob-number {
    font-family: 'DM Serif Display', serif;
    font-size: 3.6rem;
    line-height: 1;
    margin: 6px 0 2px;
}
.prob-low    { color: #059669; }
.prob-medium { color: #D97706; }
.prob-high   { color: #DC2626; }

.prob-label {
    font-size: 0.82rem;
    color: #64748B;
    font-weight: 400;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Metric chip ── */
.metric-chip {
    display: inline-block;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 0.82rem;
    color: #1E40AF;
    font-weight: 500;
    margin: 4px;
}
.metric-value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1D4ED8;
    display: block;
}

/* ── Health rec item ── */
.rec-item {
    background: #F0FDF4;
    border-left: 4px solid #10B981;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.9rem;
    color: #1A2332;
}
.rec-item.warn {
    background: #FFFBEB;
    border-left-color: #F59E0B;
}
.rec-item.alert {
    background: #FFF1F2;
    border-left-color: #F43F5E;
}
.rec-icon { font-size: 1.1rem; margin-right: 8px; }

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #0A4F6B, #0E7490) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 44px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    cursor: pointer !important;
    box-shadow: 0 4px 16px rgba(14,116,144,0.30) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0E7490, #06B6D4) !important;
    box-shadow: 0 6px 22px rgba(14,116,144,0.40) !important;
    transform: translateY(-1px) !important;
}

/* ── Input styling ── */
.stSlider [data-baseweb="slider"] { margin-top: 4px; }
div[data-baseweb="select"] > div { border-radius: 10px !important; border-color: #CBD5E1 !important; }
input[type="number"] { border-radius: 10px !important; }

/* ── Divider ── */
hr { border-color: #E2EBF0; margin: 24px 0; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 18px;
    font-size: 0.78rem;
    color: #94A3B8;
    border-top: 1px solid #E2EBF0;
    margin-top: 32px;
}

/* ── Warning banner ── */
.disclaimer {
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 0.82rem;
    color: #1E40AF;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load pre-trained model and scaler (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading prediction engine...")
def load_engine():
    model = joblib.load("xgb_diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def load_metrics():
    # Hard-coded metrics from the original notebook
    # (or you could load from metrics.json)
    return {
        "Accuracy":  0.9661,
        "Precision": 0.8661,
        "Recall":    0.7282,
        "F1-Score":  0.7912,
        "ROC-AUC":   0.9758,
    }

# Load model, scaler, and performance metrics
model, scaler = load_engine()
perf = load_metrics()

# Feature names exactly as used during training (must match order in scaled input)
feature_names = [
    "age", "hypertension", "heart_disease", "bmi",
    "HbA1c_level", "blood_glucose_level", "gender_Male",
    "smoking_history_current", "smoking_history_ever",
    "smoking_history_former", "smoking_history_never",
    "smoking_history_not current"
]

# ─────────────────────────────────────────────
# Helper functions (all unchanged)
# ─────────────────────────────────────────────
def encode_input(age, bmi, hba1c, glucose, hypertension, heart_disease, smoking):
    """Convert user inputs to the 13-feature vector expected by the model."""
    smoking_map = {
        "never":       [0, 0, 0, 1, 0],
        "former":      [0, 0, 1, 0, 0],
        "current":     [1, 0, 0, 0, 0],
        "ever":        [0, 1, 0, 0, 0],
        "not current": [0, 0, 0, 0, 1],
    }
    smoke_vec = smoking_map.get(smoking.lower(), [0, 0, 0, 1, 0])
    return np.array([[
        age, hypertension, heart_disease, bmi, hba1c, glucose,
        1,                 # gender_Male placeholder (neutral)
        *smoke_vec
    ]])

def risk_level(prob):
    if prob < 0.35:
        return "Low Risk", "low", "🟢"
    elif prob < 0.65:
        return "Moderate Risk", "medium", "🟡"
    else:
        return "High Risk", "high", "🔴"

def gauge_chart(prob):
    pct = prob * 100
    color = "#059669" if prob < 0.35 else ("#D97706" if prob < 0.65 else "#DC2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 38, "color": color, "family": "DM Serif Display"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1",
                     "tickfont": {"size": 11, "color": "#64748B"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#F8FAFC",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "#D1FAE5"},
                {"range": [35, 65], "color": "#FEF3C7"},
                {"range": [65,100], "color": "#FEE2E2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans"},
    )
    return fig

def feature_importance_chart(importances, patient_values, feature_names, scaler):
    """Personalised feature impact = global importance × |normalised patient value|."""
    patient_sc = scaler.transform(patient_values)[0]
    impact = {
        f: abs(patient_sc[i]) * importances.get(f, 0)
        for i, f in enumerate(feature_names)
    }

    # Friendly display names
    label_map = {
        "HbA1c_level":               "HbA1c Level",
        "blood_glucose_level":        "Blood Glucose",
        "bmi":                        "BMI",
        "age":                        "Age",
        "hypertension":               "Hypertension",
        "heart_disease":              "Heart Disease",
        "smoking_history_current":    "Smoking (Current)",
        "smoking_history_former":     "Smoking (Former)",
        "smoking_history_never":      "Smoking (Never)",
        "smoking_history_ever":       "Smoking (Ever)",
        "smoking_history_not current":"Smoking (Not Current)",
        "gender_Male":                "Gender",
    }

    top = sorted(impact.items(), key=lambda x: x[1], reverse=True)[:7]
    labels = [label_map.get(k, k) for k, _ in top]
    values = [v for _, v in top]
    max_v  = max(values) if values else 1

    bar_colors = []
    for v in values:
        ratio = v / max_v
        if ratio >= 0.65:
            bar_colors.append("#EF4444")
        elif ratio >= 0.35:
            bar_colors.append("#F59E0B")
        else:
            bar_colors.append("#10B981")

    fig = go.Figure(go.Bar(
        x=values[::-1],
        y=labels[::-1],
        orientation="h",
        marker=dict(color=bar_colors[::-1], line=dict(width=0)),
        text=[f"{v:.3f}" for v in values[::-1]],
        textposition="outside",
        textfont=dict(size=11, color="#475569"),
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                   showticklabels=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#334155")),
        font=dict(family="DM Sans"),
    )
    return fig

def health_recommendations(bmi, glucose, hba1c, hypertension, heart_disease, risk_label):
    recs = []
    if hba1c >= 6.5:
        recs.append(("alert", "🩸", f"HbA1c of {hba1c}% meets the diagnostic threshold for diabetes. Immediate clinical evaluation is strongly recommended."))
    elif hba1c >= 5.7:
        recs.append(("warn", "⚠️", f"HbA1c of {hba1c}% falls in the pre-diabetic range (5.7–6.4%). Consider lifestyle modifications and regular monitoring."))

    if glucose >= 200:
        recs.append(("alert", "🔬", f"Blood glucose of {glucose} mg/dL is critically elevated. Clinical blood glucose screening and specialist referral are advised."))
    elif glucose >= 126:
        recs.append(("warn", "📊", f"Fasting glucose of {glucose} mg/dL indicates potential diabetes. Follow up with a fasting plasma glucose test."))
    elif glucose >= 100:
        recs.append(("warn", "📋", f"Blood glucose of {glucose} mg/dL is in the pre-diabetic range. Dietary adjustments and physical activity are recommended."))

    if bmi >= 30:
        recs.append(("warn", "⚖️", f"BMI of {bmi:.1f} kg/m² is in the obese range. A structured weight management programme with a registered dietitian may help reduce risk."))
    elif bmi >= 25:
        recs.append(("rec", "🥗", f"BMI of {bmi:.1f} kg/m² indicates overweight. Moderate dietary changes and 150 min/week of aerobic exercise are recommended."))

    if hypertension:
        recs.append(("warn", "💊", "Hypertension is present. Blood pressure monitoring and medication adherence are important co-management strategies."))

    if heart_disease:
        recs.append(("alert", "❤️", "Pre-existing heart disease significantly elevates cardiovascular and metabolic risk. Regular cardiology and endocrinology follow-up is essential."))

    if not recs:
        recs.append(("rec", "✅", "No critical risk factors detected. Maintain a balanced diet, regular physical activity, and annual screening."))

    if risk_label == "High Risk":
        recs.append(("alert", "🏥", "Overall risk profile is HIGH. Please consult a healthcare professional for a comprehensive metabolic assessment."))

    return recs

# Extract feature importances from the pre-trained model
importances = dict(zip(feature_names, model.feature_importances_))

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <div class="header-icon">🩺</div>
  <div>
    <p class="header-title">Diabetes Risk Prediction System</p>
    <p class="header-sub">Clinical Decision Support · XGBoost + SMOTE · Trained on 96,128 Records · For Educational Use Only</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Disclaimer:</strong> This tool is developed for academic purposes only and does not constitute
  medical advice. All predictions should be interpreted by qualified healthcare professionals.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Layout: two columns
# ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.55], gap="large")

# ══════════════════════════════════════════════
# LEFT — Section 1: User Input Panel
# ══════════════════════════════════════════════
with col_left:
    st.markdown('<p class="section-label">Section 01</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Patient Input Panel</p>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        age = st.slider("🎂 Age (years)", min_value=1, max_value=100, value=45, step=1)

        bmi = st.number_input(
            "⚖️ BMI — Body Mass Index (kg/m²)",
            min_value=10.0, max_value=70.0, value=27.5, step=0.1, format="%.1f"
        )

        hba1c = st.slider(
            "🩸 HbA1c Level (%)",
            min_value=3.5, max_value=9.0, value=5.5, step=0.1, format="%.1f"
        )

        glucose = st.slider(
            "💉 Blood Glucose Level (mg/dL)",
            min_value=80, max_value=300, value=120, step=1
        )

        hypertension = st.selectbox(
            "🫀 Hypertension", options=["No", "Yes"],
            help="Does the patient have a diagnosis of hypertension?"
        )

        heart_disease = st.selectbox(
            "❤️ Heart Disease", options=["No", "Yes"],
            help="Does the patient have a prior diagnosis of heart disease?"
        )

        smoking = st.selectbox(
            "🚬 Smoking History",
            options=["Never", "Former", "Current", "Ever", "Not Current"],
            help="Patient's self-reported smoking history."
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Section 2: Predict button ──
    st.markdown('<p class="section-label" style="margin-top:8px;">Section 02</p>', unsafe_allow_html=True)
    predict_clicked = st.button("🔍  Predict Risk", use_container_width=True)

    # ── Model performance summary (always visible) ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m4, m5     = st.columns(2)
    with m1:
        st.markdown(f'<div class="metric-chip">Accuracy<span class="metric-value">{perf["Accuracy"]:.3f}</span></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-chip">F1-Score<span class="metric-value">{perf["F1-Score"]:.3f}</span></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-chip">ROC-AUC<span class="metric-value">{perf["ROC-AUC"]:.3f}</span></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-chip">Precision (Pos)<span class="metric-value">{perf["Precision"]:.3f}</span></div>', unsafe_allow_html=True)
    with m5:
        st.markdown(f'<div class="metric-chip">Recall (Pos)<span class="metric-value">{perf["Recall"]:.3f}</span></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# RIGHT — Sections 3, 4, 5
# ══════════════════════════════════════════════
with col_right:

    if not predict_clicked:
        # Placeholder state
        st.markdown("""
        <div class="card" style="text-align:center; padding:60px 40px; min-height:420px;">
          <div style="font-size:64px; margin-bottom:16px;">🩺</div>
          <p style="font-size:1.15rem; color:#64748B; font-weight:500;">
            Complete the patient form and click<br><strong style="color:#0E7490;">Predict Risk</strong> to generate results.
          </p>
          <p style="font-size:0.82rem; color:#94A3B8; margin-top:12px;">
            Risk level · Probability gauge · Feature impact · Health recommendations
          </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Encode & predict ──
        hyp_val  = 1 if hypertension == "Yes" else 0
        hd_val   = 1 if heart_disease == "Yes" else 0
        x_input  = encode_input(age, bmi, hba1c, glucose, hyp_val, hd_val, smoking)
        prob     = model.predict_proba(scaler.transform(x_input))[0][1]
        label, level, emoji = risk_level(prob)

        badge_class = {"low": "risk-low", "medium": "risk-medium", "high": "risk-high"}[level]
        prob_class  = {"low": "prob-low", "medium": "prob-medium", "high": "prob-high"}[level]

        # ── Section 3: Prediction Output ──
        st.markdown('<p class="section-label">Section 03</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1.1], gap="medium")

        with r1:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:28px 20px;">
              <span class="risk-badge {badge_class}">{emoji} {label}</span>
              <div class="prob-number {prob_class}">{prob*100:.1f}%</div>
              <div class="prob-label">Diabetes Probability</div>
              <hr style="margin:16px 0; border-color:#F1F5F9;">
              <div style="font-size:0.8rem; color:#64748B; line-height:1.6;">
                <b style="color:#1A2332;">Thresholds</b><br>
                🟢 Low &lt; 35% &nbsp;|&nbsp; 🟡 Moderate 35–65%<br>🔴 High &gt; 65%
              </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="card" style="padding:14px 16px;">', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(prob), use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 4: Feature Importance ──
        st.markdown('<p class="section-label" style="margin-top:10px;">Section 04</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Why This Prediction?</p>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="padding:20px 24px;">', unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:0.88rem; color:#475569; margin-bottom:12px;">'
            f'Personalised feature impact for this patient — '
            f'combining XGBoost feature importance with the magnitude of each input value. '
            f'<span style="color:#EF4444;">■</span> High &nbsp;'
            f'<span style="color:#F59E0B;">■</span> Moderate &nbsp;'
            f'<span style="color:#10B981;">■</span> Low</p>',
            unsafe_allow_html=True
        )
        st.plotly_chart(
            feature_importance_chart(importances, x_input, feature_names, scaler),
            use_container_width=True,
            config={"displayModeBar": False}
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Section 5: Health Recommendations ──
        st.markdown('<p class="section-label" style="margin-top:10px;">Section 05</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Health Recommendations</p>', unsafe_allow_html=True)

        recs = health_recommendations(bmi, glucose, hba1c, hyp_val, hd_val, label)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        for rec_type, icon, text in recs:
            css_class = "rec-item alert" if rec_type == "alert" else ("rec-item warn" if rec_type == "warn" else "rec-item")
            st.markdown(
                f'<div class="{css_class}"><span class="rec-icon">{icon}</span>{text}</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Diabetes Risk Prediction System &nbsp;·&nbsp;
  XGBoost · SMOTE Oversampling · sklearn + xgboost &nbsp;·&nbsp;
  Academic Use Only — Not a substitute for professional medical advice
</div>
""", unsafe_allow_html=True)