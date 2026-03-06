import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="AI Salary Prediction Dashboard", layout="wide")

# ============================================================
# Base directory
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Safe loaders
# ============================================================
def load_pickle(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(f"Missing file: {filename}")
        st.stop()
    return joblib.load(path)

def load_csv_optional(filename):
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ============================================================
# Load artifacts
# ============================================================
model = load_pickle("salary_model.pkl")      # tuned XGBoost pipeline
skill_cols = load_pickle("skill_cols.pkl")
rmse = float(load_pickle("rmse.pkl"))

df_model = load_csv_optional("df_model.csv")
results_df = load_csv_optional("results_df.csv")

# ============================================================
# Session state
# ============================================================
RESET_KEYS = [
    "age_text", "exp_text", "usd_to_inr",
    "gender_sel", "edu_sel", "job_sel",
    "skills_selected",
    "pred", "low", "high", "pred_inr", "low_inr", "high_inr",
    "monthly_usd", "monthly_inr",
    "recognized", "ignored",
    "did_predict"
]

if "do_reset" not in st.session_state:
    st.session_state.do_reset = False

if st.session_state.do_reset:
    for k in RESET_KEYS:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.do_reset = False
    st.rerun()

if "did_predict" not in st.session_state:
    st.session_state.did_predict = False

# ============================================================
# Styling
# ============================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#eaf3ff 0%,#d8ecff 45%,#eef7ff 100%);
}

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#0b2a66,#1d4ed8);
    color:white;
    border-right:2px solid rgba(255,255,255,0.15);
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown{
    color:white !important;
    font-weight:800 !important;
}

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea{
    background:white !important;
    color:black !important;
    font-size:16px !important;
    font-weight:700 !important;
    border-radius:10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] div[data-baseweb="base-input"] > div{
    background:white !important;
    border-radius:10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] span{
    color:black !important;
    font-size:16px !important;
    font-weight:700 !important;
}

section[data-testid="stSidebar"] button{
    background:linear-gradient(90deg,#3b82f6,#60a5fa) !important;
    color:white !important;
    font-weight:900 !important;
    border-radius:12px !important;
    width:100% !important;
    border:none !important;
}

.block{
    background:white;
    border-radius:18px;
    padding:20px;
    box-shadow:0 10px 28px rgba(0,0,0,0.08);
}

.kpi{
    background:white;
    border-radius:14px;
    padding:16px;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
    min-height:110px;
}

.kpi .label{
    font-size:13px;
    color:#475569;
    font-weight:800;
    margin-bottom:8px;
}

.kpi .value{
    font-size:22px;
    font-weight:900;
    color:#0f172a;
}

.small{
    color:#475569;
    font-size:13px;
}

.okbox{
    background: linear-gradient(180deg,#dcfce7,#bbf7d0);
    border:2px solid #22c55e;
    border-radius:16px;
    padding:18px;
    font-size:17px;
    font-weight:700;
}

.warnbox{
    background: linear-gradient(180deg,#fffbeb,#fde68a);
    border:2px solid #f59e0b;
    border-radius:16px;
    padding:18px;
    font-size:17px;
    font-weight:700;
}

.note{
    background:#fff7ed;
    border-left:5px solid #f97316;
    padding:12px 14px;
    border-radius:10px;
    color:#7c2d12;
    font-size:14px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Helper functions
# ============================================================
SKILL_CANONICAL = {
    "sql": "SQL",
    "api": "APIs",
    "apis": "APIs",
    "power bi": "Power BI",
    "powerbi": "Power BI",
    "ml": "Machine Learning",
    "ai": "AI",
    "dl": "Deep Learning"
}

def normalize_job_title(t):
    t = str(t).strip()
    t = re.sub(r"\s+", " ", t)
    return t

def normalize_skill_name(s):
    s2 = str(s).strip()
    low = s2.lower()
    return SKILL_CANONICAL.get(low, s2)

def plot_salary_vs_experience(df):
    tmp = df.copy()
    tmp["Years of Experience"] = tmp["Years of Experience"].round().astype(int)
    grp = tmp.groupby("Years of Experience")["Salary"].median().reset_index()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(grp["Years of Experience"], grp["Salary"], linewidth=2)
    ax.set_title("Median Salary vs Experience")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary (USD)")
    plt.tight_layout()
    return fig

def plot_avg_salary_by_role(df):
    grp = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(grp.index, grp.values)
    ax.set_title("Top 10 Average Salary by Role")
    ax.set_xlabel("Job Title")
    ax.set_ylabel("Average Salary (USD)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig

def plot_model_rmse(results_df):
    temp = results_df.copy()

    if "val_RMSE" in temp.columns:
        y_col = "val_RMSE"
        title = "Validation RMSE Comparison"
    elif "RMSE" in temp.columns:
        y_col = "RMSE"
        title = "RMSE Comparison"
    else:
        return None

    model_col = "model" if "model" in temp.columns else "Model"

    temp = temp.sort_values(y_col)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(temp[model_col], temp[y_col])
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig

def plot_model_r2(results_df):
    temp = results_df.copy()

    if "val_R2" in temp.columns:
        y_col = "val_R2"
        title = "Validation R² Comparison"
    elif "R2" in temp.columns:
        y_col = "R2"
        title = "R² Comparison"
    else:
        return None

    model_col = "model" if "model" in temp.columns else "Model"

    temp = temp.sort_values(y_col, ascending=False)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(temp[model_col], temp[y_col])
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("R²")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig

# ============================================================
# Header
# ============================================================
st.title("AI Salary Prediction Dashboard")
st.write(
    '<div class="small">Predict annual salary in USD and view INR conversion, insights, and model analytics after prediction.</div>',
    unsafe_allow_html=True
)

# ============================================================
# Sidebar Inputs
# ============================================================
with st.sidebar:
    st.subheader("User Inputs")

    gender_options = ["Select Gender", "Male", "Female", "Other"]
    edu_options = ["Select Education", "High School", "Bachelor", "Master", "PhD"]

    if df_model is not None and "Job Title" in df_model.columns:
        job_titles = sorted(df_model["Job Title"].dropna().unique())
        job_options = ["Select Job Title"] + job_titles
    else:
        job_options = ["Select Job Title"]

    with st.form("predict_form"):
        age_text = st.text_input("Age", key="age_text")
        gender_sel = st.selectbox("Gender", gender_options, key="gender_sel")
        edu_sel = st.selectbox("Education", edu_options, key="edu_sel")
        job_sel = st.selectbox("Job Title", job_options, key="job_sel")
        exp_text = st.text_input("Years of Experience", key="exp_text")

        st.multiselect(
            "Select Skills (multiple)",
            options=sorted(skill_cols),
            key="skills_selected"
        )

        usd_to_inr = st.number_input(
            "USD → INR Rate",
            value=83.0,
            min_value=1.0,
            key="usd_to_inr"
        )

        predict_btn = st.form_submit_button("Predict")
        reset_btn = st.form_submit_button("Reset")

    if reset_btn:
        st.session_state.do_reset = True
        st.rerun()

# ============================================================
# Prediction logic
# ============================================================
if predict_btn:
    error_msg = None

    try:
        age_val = int(age_text)
        exp_val = float(exp_text)
    except:
        error_msg = "Please enter valid numeric values for Age and Years of Experience."

    if gender_sel == "Select Gender":
        error_msg = "Please select Gender."
    elif edu_sel == "Select Education":
        error_msg = "Please select Education."
    elif job_sel == "Select Job Title":
        error_msg = "Please select Job Title."

    if error_msg:
        st.error(error_msg)
        st.session_state.did_predict = False
    else:
        skills_selected = st.session_state.get("skills_selected", [])
        skills_list = sorted(set([normalize_skill_name(s) for s in skills_selected]))

        recognized = [s for s in skills_list if s in skill_cols]
        ignored = [s for s in skills_list if s not in skill_cols]

        row = {
            "Age": float(age_val),
            "Years of Experience": float(exp_val),
            "Gender": gender_sel,
            "Education Level": edu_sel,
            "Job Title": normalize_job_title(job_sel)
        }

        for sc in skill_cols:
            row[sc] = 1 if sc in recognized else 0

        X_user = pd.DataFrame([row])

        pred = float(model.predict(X_user)[0])
        low = max(0, pred - rmse)
        high = pred + rmse

        rate = float(usd_to_inr)
        pred_inr = pred * rate
        low_inr = low * rate
        high_inr = high * rate

        monthly_usd = pred / 12
        monthly_inr = pred_inr / 12

        st.session_state.pred = pred
        st.session_state.low = low
        st.session_state.high = high
        st.session_state.pred_inr = pred_inr
        st.session_state.low_inr = low_inr
        st.session_state.high_inr = high_inr
        st.session_state.monthly_usd = monthly_usd
        st.session_state.monthly_inr = monthly_inr
        st.session_state.recognized = recognized
        st.session_state.ignored = ignored
        st.session_state.did_predict = True

# ============================================================
# Before prediction: show only instructions
# ============================================================
if not st.session_state.did_predict:
    st.markdown("""
    <div class="block">
        <h3 style="margin-bottom:8px;">How to use</h3>
        <div class="small">
            Fill in the user details from the sidebar and click <b>Predict</b>.<br><br>
            After prediction, the app will display:
            <ul>
                <li>Predicted annual salary in USD</li>
                <li>Approximate INR conversion</li>
                <li>Prediction range based on model RMSE</li>
                <li>Recognized skills used by the model</li>
                <li>Analytics and model comparison charts</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# After prediction: show results + charts
# ============================================================
if st.session_state.did_predict:
    pred = st.session_state.pred
    low = st.session_state.low
    high = st.session_state.high
    pred_inr = st.session_state.pred_inr
    low_inr = st.session_state.low_inr
    high_inr = st.session_state.high_inr
    monthly_usd = st.session_state.monthly_usd
    monthly_inr = st.session_state.monthly_inr
    recognized = st.session_state.recognized
    ignored = st.session_state.ignored

    left_col, right_col = st.columns([1.65, 1.0])

    # ---------------- Left: analytics tabs ----------------
    with left_col:
        st.markdown('<div class="block"><h3>Analytics & Visualization</h3><div class="small">Charts are shown only after prediction.</div></div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Market Trends", "Interpretability", "Model Comparison"])

        with tab1:
            if df_model is not None:
                st.pyplot(plot_salary_vs_experience(df_model))
                st.pyplot(plot_avg_salary_by_role(df_model))
            else:
                st.info("Dataset charts are unavailable because df_model.csv was not found.")

        with tab2:
            st.markdown("""
            <div class="note">
            From the final model analysis, <b>Years of Experience</b>, <b>Job Title</b>,
            <b>Education Level</b>, and <b>Age</b> were the most influential factors.
            Individual skills contributed, but with smaller impact than experience and role.
            </div>
            """, unsafe_allow_html=True)

            recog_text = ", ".join(recognized) if recognized else "None"
            ign_text = ", ".join(ignored) if ignored else "None"

            st.markdown(f"""
            <div class="okbox">
            ✅ Recognized Skills<br><br>
            {recog_text}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="warnbox" style="margin-top:14px;">
            ⚠️ Ignored Skills<br><br>
            {ign_text}
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            if results_df is not None:
                rmse_fig = plot_model_rmse(results_df)
                r2_fig = plot_model_r2(results_df)

                if rmse_fig is not None:
                    st.pyplot(rmse_fig)
                if r2_fig is not None:
                    st.pyplot(r2_fig)

                st.dataframe(results_df, use_container_width=True)
            else:
                st.info("Model comparison table is unavailable because results_df.csv was not found.")

    # ---------------- Right: prediction cards ----------------
    with right_col:
        st.markdown('<div class="block"><h3>Prediction Result</h3><div class="small">Results are shown after prediction.</div></div>', unsafe_allow_html=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            st.markdown(f"""
            <div class="kpi">
                <div class="label">Predicted Annual (USD)</div>
                <div class="value">${pred:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with r1c2:
            st.markdown(f"""
            <div class="kpi">
                <div class="label">Range (USD)</div>
                <div class="value">${low:,.0f} – ${high:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with r1c3:
            st.markdown(f"""
            <div class="kpi">
                <div class="label">Monthly (USD)</div>
                <div class="value">${monthly_usd:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            st.markdown(f"""
            <div class="kpi">
                <div class="label">Approx Annual (INR)</div>
                <div class="value">₹{pred_inr:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2c2:
            st.markdown(f"""
            <div class="kpi">
                <div class="label">Range (INR)</div>
                <div class="value">₹{low_inr:,.0f} – ₹{high_inr:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        with r2c3:
            st.markdown(f"""
            <div class="kpi">
                <div class="label">Monthly (INR)</div>
                <div class="value">₹{monthly_inr:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        st.markdown("""
        <div class="note">
        Salary predictions are generated in <b>USD annual salary</b>. INR values are shown
        using the selected exchange rate for easier interpretation in the Indian context.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        summary_df = pd.DataFrame({
            "Metric": ["Final Model", "MAE", "RMSE", "R²"],
            "Value": ["Tuned XGBoost", "5250.20", "9081.30", "0.9646"]
        })
        st.dataframe(summary_df, use_container_width=True)

