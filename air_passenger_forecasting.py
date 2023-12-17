import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('air_passenger_forecasting.sav', 'rb'))

# Ubah 'Month' menjadi tipe data datetime dan pastikan 'Passengers_Stationary_2' adalah numerik
df = pd.read_csv("AirPassengers.csv")
df['Month'] = pd.to_datetime(df['Month'])
df['Passengers_Stationary_2'] = pd.to_numeric(df['Passengers_Stationary_2'], errors='coerce')
df.set_index('Month', inplace=True)

st.title('Forecasting Air Passenger')
num_years = st.slider("Tentukan Jumlah Tahun untuk Forecast", 1, 10, step=1)

if st.button("Predict"):
    # Lakukan prediksi menggunakan model untuk periode yang dipilih
    forecast = model.forecast(steps=num_years * 12)  # Ubah ke bulan jika model.forecast() membutuhkan periode bulanan
    
    # Buat DataFrame untuk hasil prediksi
    pred_index = pd.date_range(start=df.index[-1], periods=num_years * 12 + 1, freq='M')[1:]
    pred = pd.DataFrame(forecast, index=pred_index, columns=['Passengers_Stationary_2'])

    # Plot hasil prediksi
    fig, ax = plt.subplots()
    df['Passengers_Stationary_2'].plot(style='--', color='gray', legend=True, label='known')
    pred['Passengers_Stationary_2'].plot(color='b', legend=True, label='Prediction')
    plt.xlabel('Month')
    plt.ylabel('Passengers_Stationary_2')
    st.pyplot(fig)
