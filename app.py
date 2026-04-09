import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Cirrhosis Outcome Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM DARK THEME CSS
# =========================================================
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0b1220 0%, #111827 45%, #1f2937 100%);
        color: #f3f4f6;
    }

    /* Main container spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Headings */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #f9fafb;
        margin-bottom: 0.3rem;
        letter-spacing: 0.3px;
    }

    .sub-text {
        font-size: 1.05rem;
        color: #cbd5e1;
        margin-bottom: 1.4rem;
        line-height: 1.6;
    }

    .section-heading {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.2rem 1.2rem 1rem 1.2rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(17,24,39,0.95));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.30);
        margin-top: 0.5rem;
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .small-note {
        font-size: 0.92rem;
        color: #cbd5e1;
        line-height: 1.6;
    }

    /* Buttons */
    div.stButton > button,
    div.stFormSubmitButton > button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 6px 16px rgba(37,99,235,0.35);
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        color: white !important;
    }

    /* Inputs - container */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stNumberInput > div > div,
    .stTextInput > div > div {
        background-color: rgba(255,255,255,0.04) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
    }

    /* Typed input text */
    input, textarea {
        color: #38bdf8 !important;
        font-weight: 600 !important;
        -webkit-text-fill-color: #38bdf8 !important;
    }

    /* Number input text */
    .stNumberInput input {
        color: #38bdf8 !important;
        font-weight: 600 !important;
        -webkit-text-fill-color: #38bdf8 !important;
    }

    /* Closed selectbox selected value */
    div[data-baseweb="select"] * {
        color: #38bdf8 !important;
        -webkit-text-fill-color: #38bdf8 !important;
    }

    div[data-baseweb="select"] span {
        color: #38bdf8 !important;
        font-weight: 600 !important;
        -webkit-text-fill-color: #38bdf8 !important;
    }

    /* Dropdown popup container */
    div[role="listbox"] {
        background-color: #111827 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }

    /* Dropdown options */
    div[role="option"] {
        background-color: #111827 !important;
        color: #38bdf8 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #38bdf8 !important;
    }

    div[role="option"] * {
        color: #38bdf8 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #38bdf8 !important;
    }

    /* Hovered option */
    div[role="option"]:hover {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    div[role="option"]:hover * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Selected option inside dropdown */
    div[role="option"][aria-selected="true"] {
        background-color: #1e293b !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    div[role="option"][aria-selected="true"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Labels */
    label, .stSelectbox label, .stNumberInput label {
        color: #e5e7eb !important;
        font-weight: 500;
    }

    /* Markdown text */
    .stMarkdown, p, li {
        color: #e5e7eb;
    }

    /* Expander */
    .streamlit-expanderHeader {
        color: #38bdf8 !important;
        font-weight: 600;
    }

    /* Dataframe */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Horizontal rule */
    hr {
        border: none;
        height: 1px;
        background: rgba(255,255,255,0.10);
        margin-top: 2rem;
        margin-bottom: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL AND METADATA
# =========================================================
model = joblib.load("inference_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
class_names = joblib.load("class_names.pkl")

# =========================================================
# DISPLAY TO BACKEND MAPPINGS
# =========================================================
drug_map = {
    "Placebo": "placebo",
    "D-Penicillamine": "d-penicillamine"
}

sex_map = {
    "Female": "f",
    "Male": "m"
}

yes_no_map = {
    "No": "n",
    "Yes": "y"
}

edema_map = {
    "None": "n",
    "Slight": "s",
    "Severe": "y"
}

# reverse maps for sample loading
drug_display_map = {v: k for k, v in drug_map.items()}
sex_display_map = {v: k for k, v in sex_map.items()}
yes_no_display_map = {v: k for k, v in yes_no_map.items()}
edema_display_map = {v: k for k, v in edema_map.items()}

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## App Overview")
    st.write(
        """
        This application predicts liver cirrhosis patient outcomes using the final tuned XGBoost model.
        """
    )

    st.markdown("### Outcome Classes")
    st.markdown("""
    - **C** → Alive at last follow-up  
    - **CL** → Alive after liver transplant  
    - **D** → Deceased
    """)

    st.markdown("### Model Performance")
    st.markdown("""
    - **Test Accuracy:** 85.07%  
    - **Macro Precision:** 0.730  
    - **Macro Recall:** 0.645  
    - **Macro F1-score:** 0.676  
    - **ROC-AUC:** 0.913
    """)

    st.markdown("### Important Note")
    st.info(
        "Raw laboratory values entered below are internally transformed where needed before being passed to the model."
    )

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🩺 Cirrhosis Outcome Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '''
    <div class="sub-text">
    Clinical analytics app for predicting patient outcome probabilities using the final tuned XGBoost model.
    <br><br>
    <span style="font-size:1.10rem; color:#94a3b8;">
    Exclusive made for Professor Desmond Bisandu by his students - S489801, S470785, S477668
    </span>
    </div>
    ''',
    unsafe_allow_html=True
)

# =========================================================
# SAMPLE PATIENT
# =========================================================
if "sample_loaded" not in st.session_state:
    st.session_state.sample_loaded = False

if st.button("Load Sample Patient"):
    st.session_state.sample_loaded = True

sample_values = {
    "N_Days": 1000,
    "Drug": "placebo",
    "Age": 50.0,
    "Sex": "f",
    "Ascites": "n",
    "Hepatomegaly": "y",
    "Spiders": "y",
    "Edema": "n",
    "Bilirubin": 1.2,
    "Cholesterol": 220.0,
    "Albumin": 3.4,
    "Copper": 55.0,
    "Alk_Phos": 1050.0,
    "SGOT": 95.0,
    "Tryglicerides": 140.0,
    "Platelets": 250.0,
    "Prothrombin": 10.5,
    "Stage": 3
}

# =========================================================
# INPUT SECTION
# =========================================================
st.markdown('<div class="section-heading">Patient Data Entry</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics & Treatment")
        n_days = st.number_input(
            "N_Days",
            min_value=0,
            value=sample_values["N_Days"] if st.session_state.sample_loaded else 1000,
            help="Number of recorded follow-up days."
        )

        drug_display = st.selectbox(
            "Drug",
            list(drug_map.keys()),
            index=list(drug_map.keys()).index(drug_display_map[sample_values["Drug"]]) if st.session_state.sample_loaded else 0
        )

        age = st.number_input(
            "Age (years)",
            min_value=0.0,
            max_value=120.0,
            value=sample_values["Age"] if st.session_state.sample_loaded else 50.0
        )

        sex_display = st.selectbox(
            "Sex",
            list(sex_map.keys()),
            index=list(sex_map.keys()).index(sex_display_map[sample_values["Sex"]]) if st.session_state.sample_loaded else 0
        )

        stage = st.selectbox(
            "Stage",
            [1, 2, 3, 4],
            index=[1, 2, 3, 4].index(sample_values["Stage"]) if st.session_state.sample_loaded else 0,
            help="Disease stage treated as an ordinal numeric predictor."
        )

    with col2:
        st.subheader("Clinical Signs")

        ascites_display = st.selectbox(
            "Ascites",
            list(yes_no_map.keys()),
            index=list(yes_no_map.keys()).index(yes_no_display_map[sample_values["Ascites"]]) if st.session_state.sample_loaded else 0
        )

        hepatomegaly_display = st.selectbox(
            "Hepatomegaly",
            list(yes_no_map.keys()),
            index=list(yes_no_map.keys()).index(yes_no_display_map[sample_values["Hepatomegaly"]]) if st.session_state.sample_loaded else 0
        )

        spiders_display = st.selectbox(
            "Spiders",
            list(yes_no_map.keys()),
            index=list(yes_no_map.keys()).index(yes_no_display_map[sample_values["Spiders"]]) if st.session_state.sample_loaded else 0
        )

        edema_display = st.selectbox(
            "Edema",
            list(edema_map.keys()),
            index=list(edema_map.keys()).index(edema_display_map[sample_values["Edema"]]) if st.session_state.sample_loaded else 0,
            help="None = no edema, Slight = slight edema, Severe = edema despite treatment"
        )

        bilirubin = st.number_input(
            "Bilirubin",
            min_value=0.0,
            value=sample_values["Bilirubin"] if st.session_state.sample_loaded else 1.0
        )

    with col3:
        st.subheader("Laboratory Values")
        cholesterol = st.number_input(
            "Cholesterol",
            min_value=0.0,
            value=sample_values["Cholesterol"] if st.session_state.sample_loaded else 200.0
        )
        albumin = st.number_input(
            "Albumin",
            min_value=0.0,
            value=sample_values["Albumin"] if st.session_state.sample_loaded else 3.5
        )
        copper = st.number_input(
            "Copper",
            min_value=0.0,
            value=sample_values["Copper"] if st.session_state.sample_loaded else 50.0
        )
        alk_phos = st.number_input(
            "Alk_Phos",
            min_value=0.0,
            value=sample_values["Alk_Phos"] if st.session_state.sample_loaded else 1000.0
        )
        sgot = st.number_input(
            "SGOT",
            min_value=0.0,
            value=sample_values["SGOT"] if st.session_state.sample_loaded else 100.0
        )
        tryglicerides = st.number_input(
            "Tryglicerides",
            min_value=0.0,
            value=sample_values["Tryglicerides"] if st.session_state.sample_loaded else 150.0
        )
        platelets = st.number_input(
            "Platelets",
            min_value=0.0,
            value=sample_values["Platelets"] if st.session_state.sample_loaded else 250.0
        )
        prothrombin = st.number_input(
            "Prothrombin",
            min_value=0.0,
            value=sample_values["Prothrombin"] if st.session_state.sample_loaded else 10.0
        )

    submitted = st.form_submit_button("Predict Outcome")

# =========================================================
# PREDICTION SECTION
# =========================================================
if submitted:
    # convert interface labels to backend values
    drug = drug_map[drug_display]
    sex = sex_map[sex_display]
    ascites = yes_no_map[ascites_display]
    hepatomegaly = yes_no_map[hepatomegaly_display]
    spiders = yes_no_map[spiders_display]
    edema = edema_map[edema_display]

    raw_input = {
        "N_Days": n_days,
        "Drug": drug_display,
        "Age": age,
        "Sex": sex_display,
        "Ascites": ascites_display,
        "Hepatomegaly": hepatomegaly_display,
        "Spiders": spiders_display,
        "Edema": edema_display,
        "Bilirubin": bilirubin,
        "Cholesterol": cholesterol,
        "Albumin": albumin,
        "Copper": copper,
        "Alk_Phos": alk_phos,
        "SGOT": sgot,
        "Tryglicerides": tryglicerides,
        "Platelets": platelets,
        "Prothrombin": prothrombin,
        "Stage": stage
    }

    model_input = {
        "N_Days": n_days,
        "Drug": drug,
        "Age": age,
        "Sex": sex,
        "Ascites": ascites,
        "Hepatomegaly": hepatomegaly,
        "Spiders": spiders,
        "Edema": edema,
        "Albumin": albumin,
        "Platelets": platelets,
        "Prothrombin": prothrombin,
        "Stage": stage
    }

    if "bilirubin_log" in feature_columns:
        model_input["bilirubin_log"] = np.log1p(bilirubin)
    if "cholesterol_log" in feature_columns:
        model_input["cholesterol_log"] = np.log1p(cholesterol)
    if "copper_log" in feature_columns:
        model_input["copper_log"] = np.log1p(copper)
    if "alk_phos_log" in feature_columns:
        model_input["alk_phos_log"] = np.log1p(alk_phos)
    if "sgot_log" in feature_columns:
        model_input["sgot_log"] = np.log1p(sgot)
    if "tryglicerides_log" in feature_columns:
        model_input["tryglicerides_log"] = np.log1p(tryglicerides)

    if "Bilirubin" in feature_columns:
        model_input["Bilirubin"] = bilirubin
    if "Cholesterol" in feature_columns:
        model_input["Cholesterol"] = cholesterol
    if "Copper" in feature_columns:
        model_input["Copper"] = copper
    if "Alk_Phos" in feature_columns:
        model_input["Alk_Phos"] = alk_phos
    if "SGOT" in feature_columns:
        model_input["SGOT"] = sgot
    if "Tryglicerides" in feature_columns:
        model_input["Tryglicerides"] = tryglicerides

    input_data = pd.DataFrame([model_input])
    input_data = input_data.reindex(columns=feature_columns)

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    predicted_class = class_names[prediction]

    prob_df = pd.DataFrame({
        "Outcome": class_names,
        "Probability": probabilities
    })
    prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)

    st.markdown('<div class="section-heading">Prediction Result</div>', unsafe_allow_html=True)
    #st.markdown('<div class="result-card">', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-label">Predicted Outcome</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{predicted_class}</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-label">Highest Probability</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{prob_df["Probability (%)"].max():.2f}%</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-label">Model Used</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">Tuned XGBoost</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if predicted_class == "C":
        st.success("The model predicts that the patient is most likely alive at the last follow-up.")
    elif predicted_class == "CL":
        st.info("The model predicts that the patient is most likely alive following liver transplant.")
    else:
        st.warning("The model predicts a higher probability of deceased outcome.")

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Prediction Probabilities")
        st.dataframe(
            prob_df[["Outcome", "Probability (%)"]],
            use_container_width=True,
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Probability Chart")
        chart_df = prob_df.set_index("Outcome")[["Probability (%)"]]
        st.bar_chart(chart_df)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("View Entered Patient Data"):
        st.dataframe(pd.DataFrame([raw_input]), use_container_width=True)

    with st.expander("View Model Input After Transformation"):
        st.dataframe(input_data, use_container_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

f1, f2 = st.columns([2, 1])

with f1:
    st.markdown("### Model Information")
    st.markdown("""
    - **Final Model:** Tuned XGBoost  
    - **Test Accuracy:** 85.07%  
    - **Macro Precision:** 0.730  
    - **Macro Recall:** 0.645  
    - **Macro F1-score:** 0.676  
    - **ROC-AUC (Macro OVR):** 0.913  
    - **Dataset:** Synthetic cirrhosis patient outcome dataset  
    """)

with f2:
    st.markdown("### Clinical Note")
    st.markdown("""
    <div class="small-note">
    This app is intended for academic demonstration only.  
    It should be treated as a decision-support prototype rather than a real clinical diagnostic system.
    This application may produce incorrect predictions, so it should not be used as the sole basis for any medical or clinical decision.
    </div>
    """, unsafe_allow_html=True)