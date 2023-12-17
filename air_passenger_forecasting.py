import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

model = pickle.load(open('air_passenger_forecasting.sav', 'rb'))

df = pd.read_csv("AirPassenger2.csv")
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df.set_index(['Month'], inplace=True)

st.title('Forecasting Kualitas Udara')
num_periods = st.slider("Tentukan Jumlah Periode untuk Diprediksi", 1, 30, step=1)

if st.button("Predict"):
    # Assuming 'Passengers_Stationary_2' is the column you want to forecast
    endog_data = df['Passengers_Stationary_2']

    # Fit ARIMA model
    arima_model = ARIMA(endog_data, order=(15, 1, 15))
    arima_result = arima_model.fit()

    # Make ARIMA forecast
    arima_forecast = arima_result.forecast(steps=num_periods)

    # Plotting known and predicted values
    fig, ax = plt.subplots()
    endog_data.plot(style='--', color='b', legend=True, label='Known')
    ax.plot(arima_forecast.index, arima_forecast.values, color='red', label='Prediction')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Passengers_Stationary_2')
    st.pyplot(fig)
