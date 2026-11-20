import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(
    page_title="Phone addiction check",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp div[data-testid="stMarkdownContainer"] {
    font-family: 'Manrope', sans-serif;
}
.block-container { max-width: 1120px; padding-top: 3.5rem; padding-bottom: 5rem; }
[data-testid="stVerticalBlock"] { gap: 1.4rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

.page-title { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; line-height: 1.1; }
.page-sub { font-size: 1.1rem; opacity: 0.72; margin: 0.6rem 0 0 0; max-width: 40rem; }
.section-title { font-size: 1.25rem; font-weight: 700; margin: 0.2rem 0 0 0; }
.section-sub { font-size: 0.92rem; opacity: 0.65; margin: 0.15rem 0 0.4rem 0; }

.result { border: 1px solid rgba(128,128,128,0.35); border-radius: 18px; padding: 2rem 2.2rem; }
.result-num { font-size: 4.2rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em; }
.result-num span { font-size: 1.5rem; font-weight: 600; opacity: 0.5; margin-left: 0.3rem; }
.result-label { font-size: 1.35rem; font-weight: 700; margin-top: 0.6rem; }
.result-text { font-size: 1rem; opacity: 0.8; margin-top: 0.5rem; max-width: 42rem; }
.scale { position: relative; height: 12px; border-radius: 999px; margin: 1.8rem 0 0.5rem 0;
         background: linear-gradient(90deg, #2a9d8f 0%, #e9c46a 40%, #f4a261 70%, #e76f51 100%); }
.scale-pin { position: absolute; top: 50%; width: 24px; height: 24px; border-radius: 50%;
             transform: translate(-50%, -50%); background: #fff; border: 4px solid #1f2933;
             box-shadow: 0 2px 8px rgba(0,0,0,0.35); }
.scale-ends { display: flex; justify-content: space-between; font-size: 0.85rem; opacity: 0.6; }
.note { font-size: 0.85rem; opacity: 0.6; margin-top: 1.2rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Load model + schema (created by the training script)
# ----------------------------------------------------------------------------
@st.cache_data
def load_schema():
    with open(BASE / "feature_schema.json") as f:
        return json.load(f)


@st.cache_resource
def load_model():
    return joblib.load(BASE / "best_addiction_model.pkl")


try:
    schema = load_schema()
except FileNotFoundError:
    st.error("feature_schema.json not found. Run the training script first.")
    st.stop()

try:
    model = load_model()
except FileNotFoundError:
    st.error("best_addiction_model.pkl not found. Run the training script first.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# ----------------------------------------------------------------------------
# Field layout: sections and friendly labels
# ----------------------------------------------------------------------------
SECTIONS = [
    ("About you", "A few basic details.",
     ["Age", "Gender"]),
    ("Phone habits", "How often and why you use your phone.",
     ["Daily_Usage_Hours", "Weekend_Usage_Hours", "Phone_Checks_Per_Day",
      "Apps_Used_Daily", "Screen_Time_Before_Bed", "Phone_Usage_Purpose"]),
    ("Screen time breakdown", "How your daily hours are split.",
     ["Time_on_Social_Media", "Time_on_Gaming", "Time_on_Education"]),
    ("Health and wellbeing", "Sleep, mood and daily life.",
     ["Sleep_Hours", "Exercise_Hours", "Anxiety_Level", "Depression_Level",
      "Self_Esteem", "Intellectual_Performance", "Social_Interactions",
      "Family_Communication"]),
]

LABELS = {
    "Daily_Usage_Hours": "Phone use per day (hours)",
    "Weekend_Usage_Hours": "Phone use on weekends (hours)",
    "Phone_Checks_Per_Day": "Times you check your phone per day",
    "Apps_Used_Daily": "Apps used per day",
    "Screen_Time_Before_Bed": "Screen time before bed (hours)",
    "Phone_Usage_Purpose": "Main reason for phone use",
    "Time_on_Social_Media": "Time on social media (hours)",
    "Time_on_Gaming": "Time on gaming (hours)",
    "Time_on_Education": "Time on learning (hours)",
    "Sleep_Hours": "Sleep per night (hours)",
    "Exercise_Hours": "Exercise per day (hours)",
    "Intellectual_Performance": "Intellectual performance",
}


def label_for(name):
    return LABELS.get(name, name.replace("_", " "))


def render_field(name, container):
    """Draw one input widget and return its value."""
    with container:
        if name in schema["categorical"]:
            return st.selectbox(label_for(name), schema["categorical"][name], key=name)

        info = schema["numeric"][name]
        lo, hi, med = info["min"], info["max"], info["median"]
        if info["is_int"]:
            lo, hi, med = int(lo), int(hi), int(round(med))
            if lo == hi:
                return st.number_input(label_for(name), value=lo, key=name)
            return st.slider(label_for(name), lo, hi, med, step=1, key=name)
        lo, hi, med = round(float(lo), 1), round(float(hi), 1), round(float(med), 1)
        if lo >= hi:
            return st.number_input(label_for(name), value=float(lo), key=name)
        return st.slider(label_for(name), lo, hi, min(max(med, lo), hi), step=0.1,
                         format="%.1f", key=name)


def level_info(pct):
    if pct < 0.25:
        return "Low", "#2a9d8f", "Your habits look balanced. Keep it up."
    if pct < 0.5:
        return "Moderate", "#c9a227", "Some signs of heavy use. Watch your late-night screen time and daily hours."
    if pct < 0.75:
        return "High", "#e08a3c", "Your habits point to heavy use. Less screen time before bed and fewer daily hours would help most."
    return "Very high", "#e76f51", "Your habits point to very heavy use. Try setting daily limits, and talk to someone you trust if it affects your day."


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
st.markdown(
    '<p class="page-title">Phone addiction check</p>'
    '<p class="page-sub">Enter your daily habits and get an estimate of your addiction level.</p>',
    unsafe_allow_html=True,
)


inputs = {}
placed = set()

with st.form("predict_form", border=False):
    all_fields = list(schema["numeric"].keys()) + list(schema["categorical"].keys())

    sections = [(t, s, [f for f in fields if f in all_fields]) for t, s, fields in SECTIONS]
    used = {f for _, _, fields in sections for f in fields}
    leftover = [f for f in all_fields if f not in used]
    if leftover:
        sections.append(("Other details", "Everything else the model uses.", leftover))

    for title, sub, fields in sections:
        if not fields:
            continue
        with st.container(border=True):
            st.markdown(f'<p class="section-title">{title}</p><p class="section-sub">{sub}</p>',
                        unsafe_allow_html=True)
            cols = st.columns(3, gap="large")
            for i, name in enumerate(fields):
                inputs[name] = render_field(name, cols[i % 3])

    submitted = st.form_submit_button("Predict addiction level", type="primary",
                                      use_container_width=True)

if submitted:
    row = dict(schema["defaults"])      # columns the model ignores
    row.update(inputs)

    # engineered features (same formulas as in the training script)
    daily = float(row.get("Daily_Usage_Hours", 0)) + 1e-5
    if "Social_Media_Ratio" in schema["engineered"]:
        row["Social_Media_Ratio"] = float(row.get("Time_on_Social_Media", 0)) / daily
        row["Gaming_Ratio"] = float(row.get("Time_on_Gaming", 0)) / daily
        row["Education_Ratio"] = float(row.get("Time_on_Education", 0)) / daily

    X_new = pd.DataFrame([row])[schema["all_columns"]]
    pred = float(model.predict(X_new)[0])
    lo, hi = schema["target_min"], schema["target_max"]
    pred = min(max(pred, lo), hi)
    pct = (pred - lo) / (hi - lo) if hi > lo else 0.0
    label, color, advice = level_info(pct)

    st.markdown(
        f"""
<div class="result">
  <div class="result-num">{pred:.1f}<span>out of {hi:g}</span></div>
  <div class="result-label" style="color:{color}">{label} addiction level</div>
  <div class="result-text">{advice}</div>
  <div class="scale"><div class="scale-pin" style="left:{pct * 100:.1f}%"></div></div>
  <div class="scale-ends"><span>{lo:g} lowest</span><span>{hi:g} highest</span></div>
  <div class="note">This is a model estimate based on your answers, not a medical diagnosis.</div>
</div>
""",
        unsafe_allow_html=True,
    )
