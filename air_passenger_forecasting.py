import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the stationary dataset (df_stationary)
df_stationary = pd.read_csv("your_stationary_dataset.csv")
# Assuming 'Passengers_Stationary_2' is the column containing the stationary data

st.title('ARIMA Forecasting')

if st.button("Predict ARIMA"):
    # Fit ARIMA model
    ar = ARIMA(df_stationary, order=(15, 1, 15)).fit()

    # Make ARIMA forecast
    ar_test_pred = ar.forecast(steps=20)

    # Plotting known and predicted values
    fig, ax = plt.subplots()
    df_stationary['Passengers_Stationary_2'].plot(style='--', color='gray', legend=True, label='Known')
    ax.plot(ar_test_pred, color='b', label='Prediction')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Passengers_Stationary_2')
    st.pyplot(fig)
