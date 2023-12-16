import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('air_passenger_forecasting.sav','rb'))

df = pd.read_csv('AirPassengers2.csv')
df['Month']=pd.to_datetime(df['Month'], format='%Y-%m-%d')
df.set_index(['Month'], inplace=True)

st.title('Forecasting Air Passenger')
month = st.slider("Tentukan Bulan",1,30, step=1)

pred = model.forecast(month)
pred = pd.DataFrame(pred, columns=['Passengers_Stationary_2'])

if st.button("Predict"):

    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(pred)
    with col2:
        fig, ax = plt.subplots()
        df_stationary['Passengers_Stationary_2'].plot(style='--', color='gray', legend=True, label='known')
        pred['Passengers_Stationary_2'].plot(color='b', legend=True, label='Prediction')
        st.pyplot(fig)
