import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('air_passenger_forecasting.sav','rb'))

df = pd.read_csv("AirPassenger2.csv")
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df.set_index(['month'], inplace=True)

st.title('Forecasting Kualitas Udara')
year = st.slider("Tentukan Tahun",1,30, step=1)

ar = ARIMA(df_stationary, order=(15,1,15)).fit()
ar_test_pred = ar.forecast(year)

if st.button("Predict"):

    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(ar)
    with col2:
        fig, ax = plt.subplots()
        df['Passengers_Stationary_2'].plot(style='--', color='gray', legend=True, label='known')
        pred['Passengers_Stationary_2'].plot(color='b', legend=True, label='Prediction')
        st.pyplot(fig)
