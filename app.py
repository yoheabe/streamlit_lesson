import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import pickle

st.title("This is my flower predictor.")
st.header('This is a great app')
st.subheader("Flowers are my fav.")

df = px.data.iris()
show_df = st.checkbox("Do you want to see the data?")
if show_df:
   df

s_l = st.number_input('Sepal Length (cm)')
s_w = st.number_input('Sepal width (cm)')
p_l = st.number_input('petal Length (cm)')
p_w = st.number_input('petal width (cm)')

user_input = np.array([s_l,s_w, p_l, p_w]).reshape(1,-1)

with open('saved-iris-model.pkl', 'rb') as flower_pickle:
   model = pickle.load(flower_pickle)
prediction = model.predict(user_input)
st.write(f'The predicted flower is {prediction[0]}')