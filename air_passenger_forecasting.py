import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open('air_passenger_forecasting.sav','rb'))

df = pd.read_csv('AirPassengers.csv')
df['Month']=pd.to_datetime(df['Month'], format='%Y-%m-%d')
df.set_index(['Month'], inplace=True)

st.title('Forecasting Air Passenger')
year = st.slider("Tentukan Tahun",1,30, step=1)

# buat DataFrame untuk data yang stasioner
df_stationary = df.copy()
# differencing 1 kali
df_stationary['Passengers_Stationary'] = df_stationary['Passengers'].diff()
# differencing 2 kali
df_stationary['Passengers_Stationary_2'] = df_stationary['Passengers'].diff().diff()

# drop baris pertama dan kedua
df_stationary = df_stationary.dropna()

del df_stationary ['Passengers']
del df_stationary ['Passengers_Stationary']
df_stationary.head()


pred = model.forecast(year)
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