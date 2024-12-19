import streamlit as st
import pandas as pd


map = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv") 

st.map(map)

widget = st.sidebar.text_input('Enter some text')

st.write(widget)    

display_analysis = st.sidebar.selectbox('Select an analysis', ['Analysis 1', 'Analysis 2', 'Analysis 3'])