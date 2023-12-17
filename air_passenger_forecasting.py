import pickle
import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load the ARIMA model
model = pickle.load(open('air_passenger_forecasting.sav', 'rb'))

# Load the dataset
df = pd.read_csv("AirPassengers.csv")
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df.set_index(['Month'], inplace=True)

st.title('Forecasting Air Passenger')
year = st.slider("Tentukan Bulan", 1, 30, step=1)

# Fit ARIMA model
ar = ARIMA(df, order=(15, 1, 15)).fit()
ar_train_pred = ar.fittedvalues

# Forecast with ARIMA
ar_test_pred = ar.forecast(steps=year)

# Convert 'Passengers_Stationary_2' column to numeric
df['Passengers_Stationary_2'] = pd.to_numeric(df['Passengers_Stationary_2'], errors='coerce')

# Split the data into train and test sets
train_df = df.loc[:'1949-03']
test_df = df.loc['1949-04':]

if st.button("Predict"):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.dataframe(ar_test_pred)
    with col2:
        fig, ax = plt.subplots()
        train_df['Passengers_Stationary_2'].plot(style='--', color='gray', legend=True, label='train_df')
        test_df['Passengers_Stationary_2'].plot(style='--', color='r', legend=True, label='test_df')
        ar_test_pred.plot(color='b', legend=True, label='Prediction')
        plt.xlabel('Month')
        plt.ylabel('Passengers_Stationary_2')
        st.pyplot(fig)
