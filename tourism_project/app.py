
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Wellness Tourism Prediction",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Wellness Tourism Purchase Prediction")

st.write(
    "Predict whether a customer is likely to purchase "
    "a wellness tourism package using our trained "
    "XGBoost machine learning model."
)


# --------------------------------------------------
# 2. Load model and deployment artifacts
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = BASE_DIR / "deployment"


@st.cache_resource
def load_model():
    return joblib.load(DEPLOYMENT_DIR / "best_model.pkl")


@st.cache_data
def load_configuration():
    with open(DEPLOYMENT_DIR / "features.json") as f:
        features = json.load(f)

    with open(DEPLOYMENT_DIR / "threshold.json") as f:
        threshold_config = json.load(f)

    return features, threshold_config


model = load_model()
feature_columns, threshold_config = load_configuration()

threshold = float(
    threshold_config["classification_threshold"]
)


# --------------------------------------------------
# 3. Customer information form
# --------------------------------------------------

st.subheader("Customer Information")

with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    contact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Enquiry"]
    )

    city_tier = st.selectbox(
        "City Tier",
        [1, 2, 3]
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Free Lancer",
            "Large Business",
            "Salaried",
            "Small Business"
        ]
    )

    gender = st.selectbox(
        "Gender",
        ["Fe Male", "Female", "Male"]
    )

    persons_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1,
        max_value=20,
        value=2
    )

    property_star = st.selectbox(
        "Preferred Property Star Rating",
        [3, 4, 5]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Divorced", "Married", "Single", "Unmarried"]
    )

    trips = st.number_input(
        "Number of Trips per Year",
        min_value=0,
        max_value=30,
        value=3
    )

    passport = st.selectbox(
        "Has Passport?",
        ["No", "Yes"]
    )

    own_car = st.selectbox(
        "Owns a Car?",
        ["No", "Yes"]
    )

    children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0,
        max_value=10,
        value=0
    )

    designation = st.selectbox(
        "Designation",
        ["AVP", "Executive", "Manager", "Senior Manager", "VP"]
    )

    income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        max_value=500000.0,
        value=25000.0,
        step=1000.0
    )

    submitted = st.form_submit_button(
        "Predict Purchase Likelihood",
        use_container_width=True
    )


# --------------------------------------------------
# 4. Prepare input dataframe and make prediction
# --------------------------------------------------

if submitted:

    customer_data = {
        "Age": float(age),
        "TypeofContact": contact,
        "CityTier": int(city_tier),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": int(persons_visiting),
        "PreferredPropertyStar": float(property_star),
        "MaritalStatus": marital_status,
        "NumberOfTrips": float(trips),
        "Passport": 1 if passport == "Yes" else 0,
        "OwnCar": 1 if own_car == "Yes" else 0,
        "NumberOfChildrenVisiting": float(children_visiting),
        "Designation": designation,
        "MonthlyIncome": float(income)
    }

    input_df = pd.DataFrame(
        [customer_data],
        columns=feature_columns
    )

    probability = float(
        model.predict_proba(input_df)[0][1]
    )

    prediction = int(probability >= threshold)

    # --------------------------------------------------
    # 5. Display prediction results
    # --------------------------------------------------

    st.subheader("Prediction Results")

    st.metric(
        "Predicted Purchase Probability",
        f"{probability:.2%}"
    )

    if prediction == 1:
        st.success(
            "Customer is predicted to purchase "
            "the wellness tourism package."
        )
    else:
        st.info(
            "Customer is predicted not to purchase "
            "the wellness tourism package."
        )

    st.caption(
        f"Classification threshold: {threshold:.2f}"
    )

    with st.expander("View Model Input Data"):
        st.dataframe(input_df, use_container_width=True)


# --------------------------------------------------
# 6. Application footer
# --------------------------------------------------

st.divider()

st.caption(
    "Visit with Us | Wellness Tourism Prediction | "
    "XGBoost + MLflow + Streamlit"
)
