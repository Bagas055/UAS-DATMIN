import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('air_passenger_forecasting.sav', 'rb'))

# Load dataset
df = pd.read_csv("AirPassengers.csv")
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

st.title('Forecasting Air Passenger')
num_years = st.slider("Tentukan Jumlah Tahun untuk Forecast", 1, 10, step=1)

if st.button("Predict"):
    # Create new index for prediction based on the frequency of the original data
    pred_index = pd.date_range(start=df.index[-1], periods=num_years * 12, freq='MS')  # MS: Month Start frequency
    
    # Perform forecast
    forecast = model.forecast(steps=num_years * 12)
    
    # Create a new DataFrame for predictions with appropriate index
    pred = pd.DataFrame(forecast, index=pred_index, columns=['Passengers_Stationary_2'])

    # Plotting known and predicted values
    fig, ax = plt.subplots()
    df['Passengers_Stationary_2'].plot(style='--', color='gray', legend=True, label='Known')
    pred.plot(ax=ax, color='b', legend=True, label='Prediction')
    plt.xlabel('Month')
    plt.ylabel('Passengers_Stationary_2')
    st.pyplot(fig)
