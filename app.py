import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained SVM model
model = joblib.load('svm_model.pkl')

# Define preprocessing pipeline based on provided steps
def preprocess_data(df):
    le = LabelEncoder()
    
    # Encode categorical features
    df['Education'] = le.fit_transform(df['Education'])
    df['City'] = le.fit_transform(df['City'])
    df['Gender'] = le.fit_transform(df['Gender'])
    df['EverBenched'] = le.fit_transform(df['EverBenched'])
    
    # Define feature columns
    feature_columns = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']
    
    # Extract features
    X = df[feature_columns]
    return X

# Streamlit app
st.title('Employee Leave Prediction')

# Define input fields
# Example data for dropdowns, replace with actual data if you have it
education_options = ['Bachelors', 'Masters', 'PhD']
city_options = ['Bangalore', 'New Delhi', 'Pune']
gender_options = ['Male', 'Female']
ever_benched_options = ['Yes', 'No']
payment_tier_options = [1, 2, 3]

# Collect user inputs
education = st.selectbox("Select Education Level", education_options)
joining_year = st.number_input("Enter Joining Year", min_value=2000, max_value=2024)
city = st.selectbox("Select City", city_options)
payment_tier = st.selectbox("Select Payment Tier", payment_tier_options)
age = st.number_input("Enter Age", min_value=18, max_value=100)
gender = st.selectbox("Select Gender", gender_options)
ever_benched = st.selectbox("Ever Benched", ever_benched_options)
experience_in_current_domain = st.number_input("Enter Experience in Current Domain (Years)", min_value=0, max_value=50)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Education': [education],
    'JoiningYear': [joining_year],
    'City': [city],
    'PaymentTier': [payment_tier],
    'Age': [age],
    'Gender': [gender],
    'EverBenched': [ever_benched],
    'ExperienceInCurrentDomain': [experience_in_current_domain]
})

# Preprocess the input data
input_data_processed = preprocess_data(input_data)

# Predict using the model
if st.button("Predict"):
    prediction = model.predict(input_data_processed)
    st.success(f"Prediction: {'Leave' if prediction[0] == 1 else 'No Leave'}")
