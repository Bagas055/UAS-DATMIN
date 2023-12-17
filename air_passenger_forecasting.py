import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('air_passenger_forecasting.sav','rb'))

df = pd.read_csv("AirPassengers.csv")
df['Year'] = pd.to_datetime(df['Passengers_Stationary_2'], format='%Y')
df.set_index(['Passengers_Stationary_2'], inplace=True)

st.title('Forecasting Air Passenger')
year = st.slider("Tentukan Tahun",1,30, step=1)

pred = model.forecast(year)
pred = pd.DataFrame(pred, columns=['CO2'])

if st.button("Predict"):

    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(pred)
    with col2:
        fig, ax = plt.subplots()
        df['Passengers_Stationary_2'].plot(style='--', color='gray', legend=True, label='known')
        pred['Passengers_Stationary_2'].plot(color='b', legend=True, label='Prediction')
        st.pyplot(fig)
