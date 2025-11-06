import numpy as np
import streamlit as st
import pandas as pd
import pickle  # to load a saved model
import base64  # optional, kept from your original import
from pathlib import Path

st.set_page_config(page_title="Loan Prediction", page_icon="💳", layout="centered")

# ---------- Helpers (Required by your request) ----------
@st.cache_data
def get_fvalue(val):
    """
    Example fixed mapping for Yes/No to numeric.
    NOTE: Keeping your original mapping (No -> 1, Yes -> 2) to demonstrate usage.
    """
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict[val]

def get_value(val, my_dict):
    """
    General dictionary lookup converter for label encoding.
    """
    return my_dict[val]

# ---------- Cached loaders ----------
@st.cache_data
def load_data(csv_path: str):
    if Path(csv_path).exists():
        return pd.read_csv(csv_path)
    # Fallback synthetic data if file not found (keeps Home tab working)
    np.random.seed(42)
    return pd.DataFrame({
        "income_annum": np.random.randint(20000, 150000, size=50),
        "loan_amount": np.random.randint(1000, 50000, size=50),
        "property_area": np.random.choice(["Urban", "Rural", "Semiurban"], size=50),
    })

@st.cache_data
def load_model(pkl_path: str):
    if Path(pkl_path).exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None  # Will trigger heuristic fallback


# ---------- Sidebar mode ----------
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Predict", "About"])

# ---------- HOME ----------
if app_mode == "Home":
    st.title("Loan Prediction")
    st.caption("Demo app with data view and charts; switch to **Predict** to try the model.")
    # Optional image if present
    if Path("chart_t1.png").exists():
        st.image("chart_t1.png", use_container_width=True)

    st.subheader("Dataset")
    data = load_data("loan_data.csv")
    st.write(data.head())

    # Simple bar chart if these columns exist
    if set(["income_annum", "loan_amount"]).issubset(data.columns):
        st.bar_chart(data[["income_annum", "loan_amount"]].head(20))

    st.write("Columns:", list(data.columns))

# ---------- PREDICT ----------
elif app_mode == "Predict":
    st.title("Loan Approval Predictor")
    st.write("Provide applicant details below and click **Predict**.")

    # Mapping dictionaries for encoding
    gender_dict = {"Male": 1, "Female": 0}
    married_dict = {"No": 0, "Yes": 1}
    education_dict = {"Graduate": 1, "Not Graduate": 0}
    self_emp_dict = {"No": 0, "Yes": 1}
    property_dict = {"Urban": 2, "Semiurban": 1, "Rural": 0}

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", list(gender_dict.keys()))
        married = st.selectbox("Married", list(married_dict.keys()))
        education = st.selectbox("Education", list(education_dict.keys()))
        self_emp = st.selectbox("Self Employed", list(self_emp_dict.keys()))
        credit_hist_yn = st.selectbox("Credit History Available?", ["Yes", "No"])  # uses get_fvalue
    with col2:
        applicant_income = st.number_input("Applicant Income (monthly)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant Income (monthly)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan Amount (total)", min_value=0, value=10000, step=500)
        loan_term = st.number_input("Loan Amount Term (months)", min_value=1, value=360, step=12)
        property_area = st.selectbox("Property Area", list(property_dict.keys()))

    # --- Encode using your two functions ---
    # Specific Yes/No mapping via get_fvalue (as you requested)
    credit_history_enc = get_fvalue(credit_hist_yn)  # {"No":1, "Yes":2}

    # General purpose encodings via get_value
    gender_enc = get_value(gender, gender_dict)
    married_enc = get_value(married, married_dict)
    education_enc = get_value(education, education_dict)
    self_emp_enc = get_value(self_emp, self_emp_dict)
    property_enc = get_value(property_area, property_dict)

    # Build feature vector (order should match training model)
    # Adjust feature names/order to your actual model’s expectation
    input_dict = {
        "Gender": gender_enc,
        "Married": married_enc,
        "Education": education_enc,
        "Self_Employed": self_emp_enc,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History_YN": credit_history_enc,  # uses get_fvalue
        "Property_Area": property_enc,
    }
    input_df = pd.DataFrame([input_dict])

    st.subheader("Encoded Input (for the model)")
    st.dataframe(input_df)

    model = load_model("loan_model.pkl")

    if st.button("Predict"):
        if model is not None:
            try:
                # Many sklearn models expect the same feature order used during training
                pred = model.predict(input_df)[0]
                proba = getattr(model, "predict_proba", lambda X: None)(input_df)
                st.success(f"Model prediction: {'Approved ✅' if int(pred)==1 else 'Not Approved ❌'}")
                if proba is not None:
                    st.caption(f"Confidence (class 1): {proba[0][1]:.2%}")
            except Exception as e:
                st.error(f"Model error: {e}")
                st.info("Using heuristic fallback instead (model feature mismatch or missing).")
                # Heuristic fallback below
                total_income = applicant_income + coapplicant_income
                ratio = (loan_amount / max(total_income, 1)) if total_income else float("inf")
                approved = (credit_history_enc == 2) and (ratio < 8) and (loan_term >= 180)
                st.write("Heuristic Decision:", "Approved ✅" if approved else "Not Approved ❌")
        else:
            # Fallback if no model file present
            total_income = applicant_income + coapplicant_income
            ratio = (loan_amount / max(total_income, 1)) if total_income else float("inf")
            approved = (credit_history_enc == 2) and (ratio < 8) and (loan_term >= 180)
            st.info("No model file found. Used a simple heuristic rule for demonstration.")
            st.write("Heuristic Decision:", "Approved ✅" if approved else "Not Approved ❌")

# ---------- ABOUT ----------
else:
    st.title("About this App")
    st.markdown(
        """
        This demo shows how to:
        - Load data and an optional trained model
        - Collect user inputs
        - **Encode features using two helper functions**:
            - `get_fvalue(val)` — fixed Yes/No mapping (cached for efficiency)
            - `get_value(val, my_dict)` — general dictionary-based encoder
        - Predict with a model (if available) or use a clear heuristic fallback

        **Why two functions?**  
        `get_fvalue` demonstrates a specific, repeated mapping for Yes/No fields,  
        while `get_value` is a reusable encoder for any categorical mapping.
        """
    )
