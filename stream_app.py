import streamlit as st 
import joblib
import pandas as pd 

model=joblib.load("customer_churn_model.pkl")

st.title("customer churn prediction")

st.write("put customer churn")


TenureMonths=st.number_input("TenureMonths",min_value=1,value=20)

TotalCharges=st.number_input("TotalCharges",min_value=1,value=10000)

SatisfactionScore=st.number_input("SatisfactionScore",min_value=1,value=5)

MonthlyCharges=st.number_input("MonthlyCharges",min_value=1,value=500)

City=st.selectbox("city",['Chennai','Ahmedabad','Bengaluru','Mumbai','Jaipur','Pune',
'Kolkata','Surat','Hyderabad','Delhi'])

DataUsageGB=st.number_input("DataUsageGB",min_value=1,value=25)

LastLoginDays=st.number_input("LastLoginDays",min_value=1,value=125)

LoginFrequency=st.number_input("LoginFrequency",min_value=1,value=12)


Age=st.number_input("Age",min_value=1,value=10)

PaymentMethod=st.selectbox("PaymentMethod",['Credit Card',
'Debit Card','Net Banking','UPI','Cash'])

if st.button("churn prediction"):
    new_customer=pd.DataFrame({
    "TenureMonths":[TenureMonths],
    "TotalCharges": [TotalCharges],
    "SatisfactionScore": [SatisfactionScore],
    "MonthlyCharges": [MonthlyCharges],
    "City": [City],
    "DataUsageGB": [DataUsageGB],
    "LastLoginDays": [LastLoginDays],
    "LoginFrequency": [LoginFrequency],
    "Age":[Age],
    "PaymentMethod":[PaymentMethod]
    })
    prediction=model.predict(new_customer)

    if prediction[0]==1:
        st.error("customer will churn")
    else:
        st.success("customer will not  churn")
