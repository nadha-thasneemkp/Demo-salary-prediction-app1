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
# Safe base directory
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Load artifacts
# ============================================================
model = joblib.load(os.path.join(BASE_DIR, "salary_model.pkl"))   # tuned XGBoost pipeline
skill_cols = joblib.load(os.path.join(BASE_DIR, "skill_cols.pkl"))
rmse = float(joblib.load(os.path.join(BASE_DIR, "rmse.pkl")))     # final tuned RMSE

df_model = None
results_df = None

try:
    df_model = pd.read_csv(os.path.join(BASE_DIR, "df_model.csv"))
except Exception:
    df_model = None

try:
    results_df = pd.read_csv(os.path.join(BASE_DIR, "results_df.csv"))
except Exception:
    results_df = None

# ============================================================
# Session reset
# ============================================================
RESET_KEYS = [
    "age_text", "exp_text", "usd_to_inr",
    "gender_sel", "edu_sel", "job_sel",
    "skills_selected",
    "pred", "low", "high", "pred_inr", "low_inr", "high_inr",
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
.stApp { background: linear-gradient(135deg,#eaf3ff 0%,#d8ecff 45%,#eef7ff 100%); }

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#0b2a66,#1d4ed8);
    color:white;
    border-right:2px solid rgba(255,255,255,0.2);
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown{
    color:white !important;
    font-weight:800 !important;
}

section[data-testid="stSidebar"] input{
    background:white !important;
    color:black !important;
    font-size:16px !important;
    font-weight:700 !important;
    border-radius:10px !important;
}

section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
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
}

.panel{
    background:white;
    border-radius:16px;
    padding:18px;
    box-shadow:0 10px 26px rgba(0,0,0,0.10);
}

.kpi{
    background:white;
    border-radius:14px;
    padding:16px;
    box-shadow:0 8px 20px rgba(0,0,0,0.08);
    text-align:center;
}

.kpi .label{ font-size:13px; color:#475569; font-weight:800; }
.kpi .value{ font-size:28px; font-weight:900; color:#0f172a; }

.small{ color:#334155; font-size:13px; }

.goodbox{
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

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(grp["Years of Experience"], grp["Salary"])
    ax.set_title("Median Salary vs Experience")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary (USD)")
    plt.tight_layout()
    return fig

def plot_avg_salary_by_role(df):
    grp = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(grp.index, grp.values)
    ax.set_title("Top 10 Average Salary by Role")
    ax.set_ylabel("Salary (USD)")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    return fig

def plot_model_rmse(results_df):
    fig, ax = plt.subplots(figsize=(7,4))
    temp = results_df.sort_values("val_RMSE")
    ax.bar(temp["model"], temp["val_RMSE"])
    ax.set_title("Validation RMSE Comparison")
    ax.set_ylabel("RMSE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig

def plot_model_r2(results_df):
    fig, ax = plt.subplots(figsize=(7,4))
    temp = results_df.sort_values("val_R2", ascending=False)
    ax.bar(temp["model"], temp["val_R2"])
    ax.set_title("Validation R² Comparison")
    ax.set_ylabel("R²")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig

def investment_recommendation(annual_salary_inr, age):
    """
    Simple rule-based investment split using INR annual salary.
    Assumes 20% of annual salary available for investment.
    """
    invest_amount = annual_salary_inr * 0.20

    if age <= 30:
        weights = {"NIFTYBEES": 0.50, "GOLDBEES": 0.25, "EMBASSY REIT": 0.25}
        risk = "High Growth"
    elif age <= 40:
        weights = {"NIFTYBEES": 0.40, "GOLDBEES": 0.30, "EMBASSY REIT": 0.30}
        risk = "Balanced"
    else:
        weights = {"NIFTYBEES": 0.25, "GOLDBEES": 0.40, "EMBASSY REIT": 0.35}
        risk = "Conservative"

    allocation = {k: invest_amount * v for k, v in weights.items()}
    return invest_amount, risk, allocation

# ============================================================
# Header
# ============================================================
st.title("AI Salary Prediction Dashboard")
st.write(
    '<div class="small">Tuned XGBoost model for salary prediction with USD → INR conversion and investment suggestions.</div>',
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
            "Select Skills",
            options=sorted(skill_cols),
            key="skills_selected"
        )

        usd_to_inr = st.number_input("USD → INR Rate", value=83.0, min_value=1.0, key="usd_to_inr")

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
        error_msg = "Please select Education Level."
    elif job_sel == "Select Job Title":
        error_msg = "Please select Job Title."

    if error_msg:
        st.error(error_msg)
        st.session_state.did_predict = False
    else:
        skills_selected = st.session_state.get("skills_selected", [])
        skills_list = sorted(set([normalize_skill_name(s) for s in skills_selected]))

        row = {
            "Age": float(age_val),
            "Years of Experience": float(exp_val),
            "Gender": gender_sel,
            "Education Level": edu_sel,
            "Job Title": normalize_job_title(job_sel)
        }

        for sc in skill_cols:
            row[sc] = 1 if sc in skills_list else 0

        X_user = pd.DataFrame([row])

        pred = float(model.predict(X_user)[0])
        low = max(0, pred - rmse)
        high = pred + rmse

        rate = float(usd_to_inr)
        pred_inr = pred * rate
        low_inr = low * rate
        high_inr = high * rate

        st.session_state.pred = pred
        st.session_state.low = low
        st.session_state.high = high
        st.session_state.pred_inr = pred_inr
        st.session_state.low_inr = low_inr
        st.session_state.high_inr = high_inr
        st.session_state.did_predict = True
        st.session_state.user_age = age_val

# ============================================================
# Main results
# ============================================================
if st.session_state.did_predict:
    pred = st.session_state.pred
    low = st.session_state.low
    high = st.session_state.high
    pred_inr = st.session_state.pred_inr
    low_inr = st.session_state.low_inr
    high_inr = st.session_state.high_inr
    user_age = st.session_state.user_age

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="kpi">
            <div class="label">Predicted Salary (USD / Year)</div>
            <div class="value">${pred:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="kpi">
            <div class="label">Predicted Salary (INR / Year)</div>
            <div class="value">₹{pred_inr:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="kpi">
            <div class="label">Model RMSE</div>
            <div class="value">{rmse:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    c4, c5 = st.columns(2)
    with c4:
        st.info(f"Estimated USD Range: **${low:,.0f} – ${high:,.0f}**")
    with c5:
        st.info(f"Estimated INR Range: **₹{low_inr:,.0f} – ₹{high_inr:,.0f}**")

    st.markdown("""
    <div class="warnbox">
    Note: Salary predictions are generated in <b>USD annual salary</b>.  
    For Indian investment recommendations, the amount is converted to <b>INR</b> using the selected exchange rate.
    </div>
    """, unsafe_allow_html=True)

    # ========================================================
    # Investment suggestion
    # ========================================================
    invest_amount, risk_profile, allocation = investment_recommendation(pred_inr, user_age)

    st.subheader("Investment Recommendation (INR-Based)")
    st.markdown(f"""
    <div class="goodbox">
    Recommended annual investment amount: <b>₹{invest_amount:,.0f}</b><br>
    Risk profile: <b>{risk_profile}</b>
    </div>
    """, unsafe_allow_html=True)

    alloc_df = pd.DataFrame({
        "Asset": list(allocation.keys()),
        "Amount (INR)": list(allocation.values())
    })

    col_a, col_b = st.columns(2)

    with col_a:
        st.dataframe(alloc_df, use_container_width=True)

    with col_b:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.pie(
            alloc_df["Amount (INR)"],
            labels=alloc_df["Asset"],
            autopct="%1.1f%%",
            startangle=90
        )
        ax.set_title("Recommended Portfolio Allocation")
        st.pyplot(fig)

# ============================================================
# Insights / graphs
# ============================================================
if df_model is not None:
    st.markdown("---")
    st.subheader("Dataset Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_salary_vs_experience(df_model))

    with col2:
        st.pyplot(plot_avg_salary_by_role(df_model))

if results_df is not None:
    st.markdown("---")
    st.subheader("Model Performance")

    c1, c2 = st.columns(2)

    with c1:
        st.pyplot(plot_model_rmse(results_df))

    with c2:
        st.pyplot(plot_model_r2(results_df))

    st.dataframe(results_df.sort_values("val_RMSE"), use_container_width=True)

# ============================================================
# Final model summary
# ============================================================
st.markdown("---")
st.subheader("Final Model Summary")

summary_df = pd.DataFrame({
    "Metric": ["Final Model", "MAE", "RMSE", "R²"],
    "Value": ["Tuned XGBoost", "5250.20", "9081.30", "0.9646"]
})

st.dataframe(summary_df, use_container_width=True)
