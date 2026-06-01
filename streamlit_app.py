import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaGuard | Water Quality AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root & Reset ── */
:root {
    --bg:        #050d1a;
    --surface:   #0b1726;
    --card:      #0f1f33;
    --border:    #1a3050;
    --accent1:   #00e5ff;
    --accent2:   #00ff9d;
    --accent3:   #7b61ff;
    --danger:    #ff4d6d;
    --text:      #e8f4fd;
    --muted:     #6b8aad;
    --glow1:     rgba(0,229,255,0.15);
    --glow2:     rgba(0,255,157,0.12);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

/* ── Main background ── */
.main .block-container {
    background: var(--bg);
    padding: 2rem 3rem;
    max-width: 1200px;
}

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #061629 0%, #0a1f3d 50%, #061629 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, var(--glow1) 0%, transparent 70%);
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -40px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, var(--glow2) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--muted);
    margin: 0;
    font-weight: 400;
    max-width: 560px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.3);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: var(--accent1);
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Stat Cards ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 140px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--accent1);
    line-height: 1;
    margin-bottom: 4px;
}
.stat-label { font-size: 0.78rem; color: var(--muted); font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Section Headers ── */
.section-hdr {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
    margin: 2rem 0 1.2rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Input Card ── */
.input-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
}
.input-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--accent1);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Streamlit Sliders & Inputs ── */
.stSlider > div > div > div { background: var(--border) !important; }
.stSlider > div > div > div > div { background: linear-gradient(90deg, var(--accent1), var(--accent2)) !important; }
div[data-baseweb="slider"] { padding: 0.3rem 0; }

/* ── Number Input ── */
div[data-baseweb="input"] > div {
    background: #0d1e30 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--accent1) !important;
    box-shadow: 0 0 0 2px rgba(0,229,255,0.15) !important;
}
div[data-baseweb="input"] input { color: var(--text) !important; font-family: 'Space Grotesk', sans-serif !important; }

/* ── Select Box ── */
div[data-baseweb="select"] > div {
    background: #0d1e30 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ── Labels ── */
label[data-testid="stWidgetLabel"] p,
.stSlider label p {
    color: #8ba8c8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Predict Button ── */
div.stButton > button {
    width: 100%;
    padding: 1rem 2rem;
    background: linear-gradient(135deg, #0066ff, #00ccff);
    color: white;
    border: none;
    border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 20px rgba(0,102,255,0.3);
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #0052cc, #00aadd);
    box-shadow: 0 6px 28px rgba(0,102,255,0.45);
    transform: translateY(-1px);
}
div.stButton > button:active { transform: translateY(0); }

/* ── Result Cards ── */
.result-potable {
    background: linear-gradient(135deg, rgba(0,255,157,0.07), rgba(0,229,255,0.05));
    border: 1px solid rgba(0,255,157,0.35);
    border-radius: 18px;
    padding: 2.2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-unsafe {
    background: linear-gradient(135deg, rgba(255,77,109,0.07), rgba(255,77,109,0.03));
    border: 1px solid rgba(255,77,109,0.35);
    border-radius: 18px;
    padding: 2.2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}
.result-sub { font-size: 0.92rem; color: var(--muted); }

/* ── Confidence Bar ── */
.conf-wrap { margin-top: 1.2rem; }
.conf-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--muted); margin-bottom: 6px; }
.conf-track { background: var(--border); border-radius: 100px; height: 8px; overflow: hidden; }
.conf-fill-safe  { height: 100%; background: linear-gradient(90deg, #00ff9d, #00e5ff); border-radius: 100px; transition: width 0.8s ease; }
.conf-fill-risk  { height: 100%; background: linear-gradient(90deg, #ff4d6d, #ff9d4d); border-radius: 100px; transition: width 0.8s ease; }

/* ── Metric Rows ── */
.metric-mini-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 1rem; }
.metric-mini {
    flex: 1; min-width: 100px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-mini-val { font-size: 1.3rem; font-weight: 700; color: var(--accent2); font-family: 'Syne', sans-serif; }
.metric-mini-lbl { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.07em; margin-top: 2px; }

/* ── WHO Reference Table ── */
.who-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.who-table th {
    background: rgba(0,229,255,0.07);
    color: var(--accent1);
    font-weight: 600;
    padding: 0.7rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.72rem;
}
.who-table td {
    padding: 0.65rem 1rem;
    border-bottom: 1px solid rgba(26,48,80,0.5);
    color: #a8c4de;
}
.who-table tr:last-child td { border-bottom: none; }
.who-table tr:hover td { background: rgba(0,229,255,0.03); }

/* ── Info Tooltip Boxes ── */
.info-box {
    background: rgba(0,229,255,0.05);
    border-left: 3px solid var(--accent1);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.2rem;
    font-size: 0.83rem;
    color: #7da8c8;
    margin-top: 1rem;
    line-height: 1.6;
}

/* ── Sidebar Nav Pills ── */
.nav-pill {
    display: block;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--muted);
    margin-bottom: 4px;
    cursor: pointer;
    transition: all 0.2s;
}
.nav-pill:hover { background: rgba(0,229,255,0.07); color: var(--accent1); }
.nav-pill.active { background: rgba(0,229,255,0.1); color: var(--accent1); font-weight: 600; }

/* ── Divider ── */
.divider { height: 1px; background: var(--border); margin: 1.5rem 0; }

/* ── Model Badge ── */
.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(123,97,255,0.12);
    border: 1px solid rgba(123,97,255,0.3);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: var(--accent3);
    font-weight: 600;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = Path("models")
    out = {}
    files = {
        "scaler": "scaler.pkl",
        "feature_cols": "feature_cols.pkl",
        "Random Forest": "Random_Forest.pkl",
        "Logistic Regression": "Logistic_Regression.pkl",
        "Decision Tree": "Decision_Tree.pkl",
        "SVM": "SVM.pkl",
        "results": "results.pkl",
    }
    for key, fname in files.items():
        p = base / fname
        if p.exists():
            out[key] = joblib.load(p)
    return out

models = load_models()
has_models = "scaler" in models and "Random Forest" in models

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0.5rem 1.5rem">
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                    background:linear-gradient(90deg,#00e5ff,#00ff9d);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;margin-bottom:0.25rem">💧 AquaGuard</div>
        <div style="font-size:0.75rem;color:#4a6f8a;font-weight:500">Water Quality Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🔍 Predict Water Quality", "📊 Model Performance", "📖 WHO Guidelines"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.72rem;color:#3a5570;line-height:1.7;padding:0 0.3rem">
        <b style="color:#4a7090">Features Used</b><br>
        9 original water parameters<br>
        + 4 engineered features<br><br>
        <b style="color:#4a7090">Models Available</b><br>
        Random Forest · SVM<br>
        Logistic Regression · Decision Tree<br><br>
        <b style="color:#4a7090">Training</b><br>
        5-Fold Stratified CV<br>
        GridSearchCV tuning
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═══════════════════════════════════════════════════════════════
if page == "🔍 Predict Water Quality":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">🤖 AI-Powered Analysis</div>
        <div class="hero-title">Water Quality<br>Classification</div>
        <p class="hero-sub">Enter water sample parameters below and get an instant potability assessment powered by machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    col_m1, col_m2 = st.columns([3, 1])
    with col_m1:
        st.markdown('<div class="section-hdr">⚗️ Input Parameters</div>', unsafe_allow_html=True)
    with col_m2:
        model_choice = st.selectbox(
            "Model",
            ["Random Forest", "Logistic Regression", "Decision Tree", "SVM"],
            index=0,
            label_visibility="collapsed"
        )

    # ── INPUT CARDS ────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="input-card"><div class="input-card-title">🧪 Chemical Parameters</div>', unsafe_allow_html=True)
        ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.01,
                       help="WHO safe range: 6.5 – 8.5")
        hardness = st.slider("Hardness (mg/L)", 50.0, 350.0, 196.0, 0.5,
                              help="Capacity of water to precipitate soap. Normal: 60–180 mg/L")
        chloramines = st.slider("Chloramines (ppm)", 0.0, 15.0, 7.1, 0.05,
                                 help="Disinfectant. WHO limit: ≤ 3 ppm")
        sulfate = st.slider("Sulfate (mg/L)", 100.0, 500.0, 333.0, 0.5,
                             help="WHO guideline: ≤ 250 mg/L")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="input-card"><div class="input-card-title">🔬 Physical Parameters</div>', unsafe_allow_html=True)
        solids = st.slider("Solids / TDS (ppm)", 300.0, 62000.0, 22014.0, 50.0,
                            help="Total Dissolved Solids. WHO limit: 500 ppm")
        conductivity = st.slider("Conductivity (μS/cm)", 100.0, 800.0, 421.0, 1.0,
                                  help="Amount of dissolved ions. WHO: ≤ 400 μS/cm")
        organic_carbon = st.slider("Organic Carbon (ppm)", 2.0, 30.0, 14.0, 0.1,
                                    help="Natural organic matter. WHO: < 2 ppm")
        trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 130.0, 66.0, 0.5,
                                     help="Disinfection byproduct. WHO: ≤ 80 μg/L")
        turbidity = st.slider("Turbidity (NTU)", 1.0, 7.0, 3.97, 0.01,
                               help="Water cloudiness. WHO: < 1 NTU ideal, ≤ 5 NTU acceptable")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ENGINEERED FEATURES PREVIEW ───────────────────────────
    ph_deviation = abs(ph - 7.0)
    mineral_load = (hardness + solids) / 2
    chem_risk = chloramines * trihalomethanes / 100
    turbidity_organic_ratio = turbidity / (organic_carbon + 1)

    with st.expander("🔧 View Engineered Features (auto-calculated)"):
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("pH Deviation", f"{ph_deviation:.3f}", help="|pH − 7.0|")
        ec2.metric("Mineral Load", f"{mineral_load:.1f}", help="(Hardness + Solids) / 2")
        ec3.metric("Chem Risk", f"{chem_risk:.3f}", help="Chloramines × THMs / 100")
        ec4.metric("Turbidity/OC Ratio", f"{turbidity_organic_ratio:.4f}", help="Turbidity / (OC + 1)")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PREDICT BUTTON ────────────────────────────────────────
    predict_btn = st.button("⚡ Analyze Water Sample", use_container_width=True)

    if predict_btn:
        if not has_models:
            st.markdown("""
            <div class="info-box">
                ⚠️ <b>Model files not found.</b> Please run the training notebook first to generate
                <code>models/</code> directory with <code>.pkl</code> files, then place them
                alongside this <code>app.py</code>.
            </div>
            """, unsafe_allow_html=True)

            # Demo mode
            st.markdown("---")
            st.markdown("**Demo Prediction (simulated)**")
            demo_prob = 0.67 if ph > 6.5 and ph < 8.5 else 0.28
            demo_label = 1 if demo_prob > 0.5 else 0
        else:
            # Build feature vector
            feature_cols = models["feature_cols"]
            row = {
                'ph': ph,
                'Hardness': hardness,
                'Solids': solids,
                'Chloramines': chloramines,
                'Sulfate': sulfate,
                'Conductivity': conductivity,
                'Organic_carbon': organic_carbon,
                'Trihalomethanes': trihalomethanes,
                'Turbidity': turbidity,
                'ph_deviation': ph_deviation,
                'mineral_load': mineral_load,
                'chem_risk': chem_risk,
                'turbidity_organic_ratio': turbidity_organic_ratio,
            }
            X_input = pd.DataFrame([row])[feature_cols]
            X_scaled = models["scaler"].transform(X_input)

            clf = models.get(model_choice)
            if clf is None:
                st.error(f"Model '{model_choice}' not found in models/ directory.")
                st.stop()

            pred = clf.predict(X_scaled)[0]
            prob = clf.predict_proba(X_scaled)[0]
            prob_potable = prob[1]
            prob_unsafe  = prob[0]

            # ── RESULT DISPLAY ────────────────────────────────
            if pred == 1:
                st.markdown(f"""
                <div class="result-potable">
                    <div class="result-icon">✅</div>
                    <div class="result-label" style="color:#00ff9d">POTABLE</div>
                    <div class="result-sub">This water sample is predicted to be <strong>safe for consumption</strong></div>
                    <div class="conf-wrap">
                        <div class="conf-label">
                            <span>Confidence</span>
                            <span style="color:#00ff9d;font-weight:700">{prob_potable*100:.1f}%</span>
                        </div>
                        <div class="conf-track">
                            <div class="conf-fill-safe" style="width:{prob_potable*100:.1f}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-unsafe">
                    <div class="result-icon">⚠️</div>
                    <div class="result-label" style="color:#ff4d6d">NOT POTABLE</div>
                    <div class="result-sub">This water sample is predicted to be <strong>unsafe for consumption</strong></div>
                    <div class="conf-wrap">
                        <div class="conf-label">
                            <span>Risk Level</span>
                            <span style="color:#ff4d6d;font-weight:700">{prob_unsafe*100:.1f}%</span>
                        </div>
                        <div class="conf-track">
                            <div class="conf-fill-risk" style="width:{prob_unsafe*100:.1f}%"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            pb1, pb2, pb3 = st.columns(3)
            pb1.metric("Potable Probability", f"{prob_potable*100:.2f}%")
            pb2.metric("Unsafe Probability", f"{prob_unsafe*100:.2f}%")
            pb3.metric("Model Used", model_choice.split()[0])

            # Parameter flags
            st.markdown('<div class="section-hdr" style="margin-top:2rem">🚨 Parameter Flags</div>', unsafe_allow_html=True)
            flags = []
            if ph < 6.5 or ph > 8.5:
                flags.append(f"**pH** ({ph:.2f}) is outside WHO range (6.5–8.5)")
            if solids > 500:
                flags.append(f"**TDS** ({solids:.0f} ppm) exceeds WHO guideline (500 ppm)")
            if chloramines > 3:
                flags.append(f"**Chloramines** ({chloramines:.2f} ppm) above WHO limit (3 ppm)")
            if sulfate > 250:
                flags.append(f"**Sulfate** ({sulfate:.0f} mg/L) above WHO guideline (250 mg/L)")
            if trihalomethanes > 80:
                flags.append(f"**Trihalomethanes** ({trihalomethanes:.1f} μg/L) above WHO limit (80 μg/L)")
            if turbidity > 5:
                flags.append(f"**Turbidity** ({turbidity:.2f} NTU) above acceptable limit (5 NTU)")

            if flags:
                for f in flags:
                    st.warning(f)
            else:
                st.success("✅ All parameters are within WHO acceptable ranges.")


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">📈 Evaluation Report</div>
        <div class="hero-title">Model<br>Performance</div>
        <p class="hero-sub">Comparison of 4 classifiers trained on the water potability dataset using 5-fold stratified cross-validation.</p>
    </div>
    """, unsafe_allow_html=True)

    if not has_models or "results" not in models:
        st.markdown("""
        <div class="info-box">
            ⚠️ Model results not found. Run the training notebook to generate <code>models/results.pkl</code>.
            <br><br>Below are the expected results based on the training notebook.
        </div>
        """, unsafe_allow_html=True)

        # Hardcoded fallback results from the notebook
        results_data = {
            "Random Forest":       {"accuracy": 0.6707, "precision": 0.6102, "recall": 0.5734, "f1": 0.5912},
            "SVM":                 {"accuracy": 0.6601, "precision": 0.5995, "recall": 0.5489, "f1": 0.5731},
            "Logistic Regression": {"accuracy": 0.5946, "precision": 0.5124, "recall": 0.3680, "f1": 0.4284},
            "Decision Tree":       {"accuracy": 0.6311, "precision": 0.5622, "recall": 0.5734, "f1": 0.5677},
        }
    else:
        results_data = {
            name: {k: res[k] for k in ['accuracy', 'precision', 'recall', 'f1']}
            for name, res in models["results"].items()
        }

    # Best model highlight
    best_name = max(results_data, key=lambda n: results_data[n]['f1'])
    best = results_data[best_name]

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-val">{best['accuracy']*100:.1f}%</div>
            <div class="stat-label">Best Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{best['precision']*100:.1f}%</div>
            <div class="stat-label">Best Precision</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{best['recall']*100:.1f}%</div>
            <div class="stat-label">Best Recall</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{best['f1']*100:.1f}%</div>
            <div class="stat-label">Best F1 Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-val" style="font-size:1.1rem">{best_name.split()[0]}</div>
            <div class="stat-label">Top Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Performance table
    st.markdown('<div class="section-hdr">📋 Detailed Scores</div>', unsafe_allow_html=True)

    table_rows = ""
    for name, r in sorted(results_data.items(), key=lambda x: -x[1]['f1']):
        crown = " 👑" if name == best_name else ""
        table_rows += f"""
        <tr>
            <td><b style="color:#c8dff0">{name}{crown}</b></td>
            <td>{r['accuracy']*100:.2f}%</td>
            <td>{r['precision']*100:.2f}%</td>
            <td>{r['recall']*100:.2f}%</td>
            <td><b style="color:#00ff9d">{r['f1']*100:.2f}%</b></td>
        </tr>"""

    st.markdown(f"""
    <div class="input-card">
    <table class="who-table">
        <thead><tr>
            <th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # Training info
    st.markdown('<div class="section-hdr">⚙️ Training Details</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="input-card">
        <table class="who-table">
            <thead><tr><th>Setting</th><th>Value</th></tr></thead>
            <tbody>
                <tr><td>Dataset</td><td>water_potability.csv — 3276 samples</td></tr>
                <tr><td>Features</td><td>13 (9 original + 4 engineered)</td></tr>
                <tr><td>Train / Test Split</td><td>80% / 20% (stratified)</td></tr>
                <tr><td>Cross-Validation</td><td>5-Fold Stratified CV</td></tr>
                <tr><td>Tuning Method</td><td>GridSearchCV</td></tr>
                <tr><td>Missing Values</td><td>Filled with per-class median</td></tr>
                <tr><td>Scaling</td><td>StandardScaler (fit on train only)</td></tr>
                <tr><td>Class Imbalance</td><td>~61% Not Potable / ~39% Potable</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — WHO GUIDELINES
# ═══════════════════════════════════════════════════════════════
elif page == "📖 WHO Guidelines":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">🌍 WHO Standards</div>
        <div class="hero-title">Water Quality<br>Guidelines</div>
        <p class="hero-sub">World Health Organization reference values for drinking water safety parameters used in this model.</p>
    </div>
    """, unsafe_allow_html=True)

    # Key stats
    st.markdown("""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-val">9</div>
            <div class="stat-label">Raw Parameters</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">4</div>
            <div class="stat-label">Engineered Features</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">13</div>
            <div class="stat-label">Total Features</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">WHO</div>
            <div class="stat-label">Reference Authority</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">📏 Original Parameters</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="input-card">
    <table class="who-table">
        <thead><tr>
            <th>Parameter</th><th>Unit</th><th>WHO Guideline</th><th>Description</th>
        </tr></thead>
        <tbody>
            <tr><td><b>pH</b></td><td>—</td><td>6.5 – 8.5</td><td>Acidity/alkalinity of water</td></tr>
            <tr><td><b>Hardness</b></td><td>mg/L</td><td>≤ 300 mg/L</td><td>Calcium & magnesium salts concentration</td></tr>
            <tr><td><b>Solids (TDS)</b></td><td>ppm</td><td>≤ 500 ppm</td><td>Total dissolved solids</td></tr>
            <tr><td><b>Chloramines</b></td><td>ppm</td><td>≤ 3 ppm</td><td>Disinfection chemical added to water</td></tr>
            <tr><td><b>Sulfate</b></td><td>mg/L</td><td>≤ 250 mg/L</td><td>Naturally occurring mineral compound</td></tr>
            <tr><td><b>Conductivity</b></td><td>μS/cm</td><td>≤ 400 μS/cm</td><td>Ability of water to conduct electricity</td></tr>
            <tr><td><b>Organic Carbon</b></td><td>ppm</td><td>< 2 ppm</td><td>Natural organic matter from decaying material</td></tr>
            <tr><td><b>Trihalomethanes</b></td><td>μg/L</td><td>≤ 80 μg/L</td><td>Byproduct of chlorination disinfection</td></tr>
            <tr><td><b>Turbidity</b></td><td>NTU</td><td>≤ 5 NTU</td><td>Cloudiness/haziness of water</td></tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">🔧 Engineered Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="input-card">
    <table class="who-table">
        <thead><tr>
            <th>Feature</th><th>Formula</th><th>Purpose</th>
        </tr></thead>
        <tbody>
            <tr>
                <td><b>pH Deviation</b></td>
                <td><code>|pH − 7.0|</code></td>
                <td>Distance from neutral pH — higher deviation = higher risk</td>
            </tr>
            <tr>
                <td><b>Mineral Load</b></td>
                <td><code>(Hardness + Solids) / 2</code></td>
                <td>Combined mineral concentration index</td>
            </tr>
            <tr>
                <td><b>Chemical Risk</b></td>
                <td><code>Chloramines × THMs / 100</code></td>
                <td>Interaction proxy for disinfection byproducts toxicity</td>
            </tr>
            <tr>
                <td><b>Turbidity/OC Ratio</b></td>
                <td><code>Turbidity / (OC + 1)</code></td>
                <td>Abnormal turbidity vs organic carbon — contamination indicator</td>
            </tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 <b>Why engineered features?</b><br>
        Raw measurements alone may miss non-linear relationships between parameters.
        Engineered features encode domain knowledge from WHO standards, capturing interactions
        that individual measurements cannot express — improving model accuracy without adding external data.
    </div>
    """, unsafe_allow_html=True)