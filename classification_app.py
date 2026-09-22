import numpy as np
import pandas as pd
import streamlit as st
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as TTS

import tensorflow as tf
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('./dl_model/classification_model.h5')

# Load label encoder
with open('./preprocessors/classification/gender_label_encoder.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

# Load onehot encoder
with open('./preprocessors/classification/onehot_geo_encoder.pkl', 'rb') as file:
    geo_encoder = pickle.load(file)

# Load standard scaler
with open('./preprocessors/classification/standard_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit App
st.title("Customer Churn Prediction using ANN")

# User input
geography = st.selectbox(label='Geography', options=geo_encoder.categories_[0])
gender = st.selectbox(label='Gender', options=gender_encoder.classes_)
age = st.slider(label='Age', min_value=10, max_value=95)
balance = st.number_input(label='Balance')
credit_score = st.number_input(label='Credit Score')
estimated_salary = st.number_input(label='Estimated Salary')
tenure = st.slider(label='Tenure', min_value=0, max_value=10)
num_of_products = st.slider(label='Number of Products', min_value=1, max_value=4)
has_cr_card = st.radio(label='Has credit card', options=[1,0], 
                       format_func=lambda x: 'Yes' if x ==1 else 'No')
is_active_member = st.radio(label='Is active member', options=[1,0],
                            format_func=lambda x: 'Yes' if x ==1 else 'No')

# Prepare input data

## For geography data
geo_df = pd.DataFrame(data = geo_encoder.transform([[geography]]), 
                      columns = geo_encoder.get_feature_names_out())
## For remaining data
input_data = pd.DataFrame({'CreditScore': [credit_score],
                           'Gender': [gender_encoder.transform([gender])[0]],
                           'Age': [age],
                           'Tenure': [tenure],
                           'Balance': [balance],
                           'NumOfProducts': [num_of_products],
                           'HasCrCard': [has_cr_card],
                           'IsActiveMember': [is_active_member],
                           'EstimatedSalary': [estimated_salary]})

# Concat both data into one
input_df = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Standard Scaling
scaled_input_df = scaler.transform(input_df)

# Predict churn data
prediction_proba = model.predict(scaled_input_df)[0][0]
st.write("Probability to churn:", prediction_proba * 100)

if prediction_proba > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")