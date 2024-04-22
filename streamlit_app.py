import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

st.title("Bank Customer Churn Prediction App")


model = joblib.load("best_model1.pkl")


credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600, step=1)
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5, step=1)
balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=10000.0, step=100.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])

is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=50000.0, step=100.0)

gender = st.selectbox("Gender", ["Male", "Female"])

if st.button("Predict"):
    
    gender_male = 1 if gender == "Male" else 0
    
    
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
        "Gender_Male": [gender_male]
    })

    
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    
    prediction = model.predict(input_data_scaled)

    
    if prediction[0] == 1:
        st.success("The customer is not at risk of churn.")
    else:
        st.success("The customer is  at risk of churn.")

