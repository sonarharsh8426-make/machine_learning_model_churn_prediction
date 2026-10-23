# Customer Churn Prediction | Machine Learning

A machine learning web application designed to predict customer churn using customer demographics, account information, satisfaction scores, usage patterns, and payment methods.

**Live Demo:** [Customer Churn Prediction App](https://machinelearningmodelchurnprediction-grh4a2zxd8dv2brdpmoeba.streamlit.app/)

## Project Overview

Customer churn is a significant business challenge that affects customer retention and revenue. This project uses a trained machine learning classification model to predict whether a customer is likely to leave a service.

The application features an interactive web interface built with Streamlit, allowing users to enter customer information and view prediction results.

## Project Objectives

- Develop a machine learning-based customer churn prediction system.
- Process and manage customer data using Pandas.
- Integrate a trained machine learning model into a web application.
- Display prediction results through an interactive user interface.
- Deploy the application for online access.

## Technology Stack

- **Python** — Core programming language
- **Pandas** — Data manipulation and processing
- **Scikit-learn** — Machine learning and classification
- **Joblib** — Saving and loading the trained model
- **Streamlit** — Interactive web application development
- **GitHub** — Source code management
- **Streamlit Community Cloud** — Application deployment

## Key Features

- Interactive customer information form
- Numeric inputs and categorical dropdown menus
- Machine learning-based churn prediction
- Color-coded prediction results
- Customer input summary
- Online application access through a web browser

## Input Features

The application uses the following customer attributes:

- Tenure Months
- Total Charges
- Satisfaction Score
- Monthly Charges
- City
- Data Usage (GB)
- Last Login Days
- Login Frequency
- Age
- Payment Method

## Project Workflow

1. **Data Preparation:** Prepare and preprocess the customer dataset for machine learning.
2. **Model Training:** Train a classification model using the prepared dataset.
3. **Model Serialization:** Save the trained model as `customer_churn_model.pkl` using Joblib.
4. **Application Development:** Build an interactive web interface using Streamlit.
5. **Customer Input:** Collect customer details through numeric inputs and dropdown menus.
6. **Prediction:** Pass the customer data to the trained model to generate a churn prediction.
7. **Result Visualization:** Display the prediction using clear, color-coded messages.
8. **Deployment:** Deploy the application using Streamlit Community Cloud for online access.
