import pandas as pd
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import joblib

model=joblib.load('car_price_pred.pkl')
scaler=joblib.load('scaler.pkl')
feature_columns=joblib.load('feature_columns.pkl')

## Taking input from user
year = st.number_input("Year", 1990, 2026, 2020)
mileage = st.number_input("Total Running", 0, 300000, 30000)
mpg = st.number_input("Mileage(MPG)", 0.0, 250.0, 40.0)
tax = st.number_input("Tax", 0, 1000, 150)
engineSize=st.number_input("Engine Size",1.0,5.0,1.5)


fuelType = st.selectbox(
    "Fuel Type",
    ["Diesel","Electric","Hybrid","Petrol","other"]
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic","Manual","Semi-Auto"]
)

car_model=st.selectbox(
    "Model",
    [' Fiesta', ' Focus', ' Puma', ' Kuga', ' EcoSport', ' C-MAX',
       ' Mondeo', ' Ka+', ' Tourneo Custom', ' S-MAX', ' B-MAX', ' Edge',
       ' Tourneo Connect', ' Grand C-MAX', ' KA', ' Galaxy', ' Mustang',
       ' Grand Tourneo Connect', ' Fusion', ' Ranger', ' Streetka',
       ' Escort', ' Transit Tourneo', 'Focus']
)

input_data = pd.DataFrame({
    'model': [car_model],
    'year': [year],
    'transmission': [transmission],
    'mileage': [mileage],
    'fuelType': [fuelType],
    'tax': [tax],
    'mpg': [mpg],
    'engineSize':[engineSize]
})

input_data = pd.get_dummies(
    input_data,
    columns=['fuelType', 'transmission', 'model']
)

input_data = input_data.reindex(
    columns=feature_columns,
    fill_value=0
)

input_scaled = scaler.transform(input_data)

if st.button("Predict Price"):

    prediction = model.predict(input_scaled)

    st.success(
        f"Predicted Car Price: £{prediction[0]:,.2f}"
    )