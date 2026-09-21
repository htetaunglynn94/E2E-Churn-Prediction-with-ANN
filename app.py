import numpy as np
import pandas as pd
import streamlit as st
import pickle

from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title = "Prediction App", layout = "wide")

st.title("ANN Deep Learning Prediction")


# ============================================================
# TABS
# ============================================================

classification_tab, regression_tab = st.tabs(["CLASSIFICATION", "REGRESSION"])


# ============================================================
# CLASSIFICATION TAB
# ============================================================

with classification_tab:

    st.header("Customer Churn Prediction")
    st.write("Predict whether a customer is likely to churn or not.")

    # Load classification model
    classification_model = load_model('./dl_model/classification_model.h5')

    # Load classification encoders
    with open('./preprocessors/classification/gender_label_encoder.pkl', 'rb') as file:
        class_gender_encoder = pickle.load(file)

    with open('./preprocessors/classification/onehot_geo_encoder.pkl', 'rb') as file:
        class_geo_encoder = pickle.load(file)

    with open('./preprocessors/classification/standard_scaler.pkl', 'rb') as file:
        class_scaler = pickle.load(file)


    # -------------------------
    # User Input
    # -------------------------

    geography = st.selectbox(label='Geography', options=class_geo_encoder.categories_[0],
                             key='class_geography')
    gender = st.selectbox(label='Gender', options=class_gender_encoder.classes_,
                          key='class_gender')
    age = st.slider(label='Age', min_value=10, max_value=95, key='class_age')
    balance = st.number_input(label='Balance ($)', key='class_balance')
    credit_score = st.number_input(label='Credit Score', key='class_credit_score')
    estimated_salary = st.number_input(label='Estimated Salary ($)')
    tenure = st.slider(label='Tenure', min_value=0, max_value=10, key='class_tenure')
    num_of_products = st.slider(label='Number of Products', min_value=1, max_value=4, key='class_nop')
    has_cr_card = st.radio(label='Has credit card?', options=[1, 0],
                           format_func=lambda x: 'Yes' if x == 1 else 'No',
                           key='class_hcc')
    is_active_member = st.radio(label='Is active member?', options=[1, 0],
                                format_func=lambda x: 'Yes' if x == 1 else 'No',
                                key='class_iam')


    # -------------------------
    # Prepare Input
    # -------------------------

    geo_df = pd.DataFrame(data = class_geo_encoder.transform([[geography]]),
                          columns = class_geo_encoder.get_feature_names_out())
    
    input_data = pd.DataFrame({'CreditScore': [credit_score],
                               'Gender': [class_gender_encoder.transform([gender])[0]],
                               'Age': [age],
                               'Tenure': [tenure],
                               'Balance': [balance],
                               'NumOfProducts': [num_of_products],
                               'HasCrCard': [has_cr_card],
                               'IsActiveMember': [is_active_member],
                               'EstimatedSalary': [estimated_salary]})

    input_df = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)


    # -------------------------
    # Prediction
    # -------------------------

    scaled_input_df = class_scaler.transform(input_df)

    prediction_proba = classification_model.predict(scaled_input_df, verbose=0)[0][0]


    # -------------------------
    # Result
    # -------------------------

    st.subheader("Prediction Result")

    st.write(f"**Probability to churn:** " 
             f"{prediction_proba * 100:.2f}%")

    if prediction_proba > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")


# ============================================================
# REGRESSION TAB
# ============================================================

with regression_tab:

    st.header("Estimated Salary Prediction")
    st.write("Predict the customer's estimated salary.")

    # Load regression model
    regression_model = load_model('./dl_model/regression_model.h5')

    # Load regression encoders
    with open('./preprocessors/regression/gender_label_encoder.pkl', 'rb') as file:
        reg_gender_encoder = pickle.load(file)

    with open('./preprocessors/regression/onehot_geo_encoder.pkl', 'rb') as file:
        reg_geo_encoder = pickle.load(file)

    with open('./preprocessors/regression/standard_scaler.pkl', 'rb') as file:
        reg_scaler = pickle.load(file)


    # -------------------------
    # User Input
    # -------------------------

    geography = st.selectbox(label='Geography', options=reg_geo_encoder.categories_[0],
                             key='reg_geography')
    gender = st.selectbox(label='Gender', options=reg_gender_encoder.classes_, key='reg_gender')
    age = st.slider(label='Age', min_value=10, max_value=95, key='reg_age')
    balance = st.number_input(label='Balance ($)', key='reg_balance')
    credit_score = st.number_input(label='Credit Score', key='reg_credit_score')
    tenure = st.slider(label='Tenure', min_value=0, max_value=10, key='reg_tenure')
    num_of_products = st.slider(label='Number of Products', min_value=1, max_value=4, key='reg_nop')
    has_cr_card = st.radio(label='Has credit card?', options=[1, 0],
                            format_func=lambda x: 'Yes' if x == 1 else 'No',
                            key='reg_hcc')
    is_active_member = st.radio(label='Is active member?', options=[1, 0],
                                format_func=lambda x: 'Yes' if x == 1 else 'No',
                                key='reg_iam')
    is_exited = st.radio(label='Is exited?', options=[1, 0],
                         format_func=lambda x: 'Yes' if x == 1 else 'No')


    # -------------------------
    # Prepare Input
    # -------------------------

    geo_df = pd.DataFrame(data=reg_geo_encoder.transform([[geography]]),
                          columns=reg_geo_encoder.get_feature_names_out())

    input_data = pd.DataFrame({'CreditScore': [credit_score],
                               'Gender': [reg_gender_encoder.transform([gender])[0]],
                               'Age': [age],
                               'Tenure': [tenure],
                               'Balance': [balance],
                               'NumOfProducts': [num_of_products],
                               'HasCrCard': [has_cr_card],
                               'IsActiveMember': [is_active_member],
                               'Exited': [is_exited]})

    input_df = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)


    # -------------------------
    # Prediction
    # -------------------------

    scaled_input_df = reg_scaler.transform(input_df)

    predicted_salary = regression_model.predict(scaled_input_df, verbose=0)[0][0]


    # -------------------------
    # Result
    # -------------------------

    st.subheader("Prediction Result")
    st.success(f"Predicted Estimated Salary: ${predicted_salary:,.2f}")