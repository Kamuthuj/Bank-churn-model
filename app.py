import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


model = joblib.load("best_model1.pkl")


st.title("Bank Customer Churn Prediction App.")


credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600, step=1)
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=10000.0, step=100.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=300000.0, value=50000.0, step=100.0)


if st.button("Predict"):

    input_data = pd.DataFrame(
        {
            "CreditScore": [credit_score],
            "Age": [age],
        
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            
            "EstimatedSalary": [estimated_salary],
            
          
        }
    )

   
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

 
    prediction = model.predict(input_data_scaled)

   
    if prediction[0] == 1:
        st.success("The customer is not at risk of churn.")
    else:
        st.success("The customer is at risk of churn.")
