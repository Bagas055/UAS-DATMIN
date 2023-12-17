import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('air_passenger_forecasting.sav','rb'))

df = pd.read_csv("AirPassengers.csv")
df['Year'] = pd.to_datetime(df['Month'], format='%Y-%m')
df.set_index(['Month'], inplace=True)

st.title('Forecasting Air Passenger')
year = st.slider("Tentukan Tahun",1,30, step=1)

ar = ARIMA(df_stationary, order=(15,1,15)).fit()
ar_train_pred = ar.fittedvalues
ar_test_pred = ar.forecast(year)

train_df_stationary = df_stationary.loc[:'1949-03']
test_df_stationary = df_stationary.loc['1949-04':]

pred = model.forecast(year)
pred = pd.DataFrame(pred, columns=['Passengers_Stationary_2'])

if st.button("Predict"):

    col1, col2 = st.columns([2,3])
    with col1:
        st.dataframe(pred)
    with col2:
        fig, ax = plt.subplots()
        train_df_stationary['Passengers_Stationary_2'].plot(style='--', color='gray', legend=True, label='train_df')
        test_df_stationary['Passengers_Stationary_2'].plot(style='--', color='r', legend=True, label='test_df')
        ar_test_pred.plot(color='b', legend=True, label='Prediction')
        st.pyplot(fig)
