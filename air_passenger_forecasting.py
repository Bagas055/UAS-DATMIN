import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('air_passenger_forecasting.sav', 'rb'))

# Ubah sesuai dengan dataset baru yang memiliki kolom 'Month' dan 'Passengers_Stationary_2'
df = pd.read_csv("AirPassengers.csv")
df['Month'] = pd.to_datetime(df['Month'])  # Pastikan kolom 'Month' memiliki tipe data datetime
df.set_index(['Month'], inplace=True)

df['Passengers_Stationary_2'] = pd.to_numeric(df['Passengers_Stationary_2'], errors='coerce')

st.title('Forecasting Air Passengers')

# Ubah slider menjadi memilih jumlah bulan untuk diprediksi
months_to_predict = st.slider("Tentukan Jumlah Bulan untuk Diprediksi", 1, 30, step=1)

if st.button("Predict"):

    # Lakukan prediksi menggunakan model yang telah diload
    pred = model.forecast(steps=months_to_predict)
    pred = pd.DataFrame(pred, columns=['Passengers_Stationary_2'])

    # Tampilkan tabel hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write(pred)

    # Buat plot menggunakan Matplotlib dan tampilkan di Streamlit
    fig, ax = plt.subplots()
    df['Passengers_Stationary_2'].plot(style='--', color='gray', legend=True, label='Known')
    pred['Passengers_Stationary_2'].plot(color='b', legend=True, label='Prediction')
    plt.xlabel('Month')
    plt.ylabel('Passengers_Stationary_2')
    st.pyplot(fig)
