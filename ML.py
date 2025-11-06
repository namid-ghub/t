import numpy as np
import streamlit as st
import pandas as pd


@st.cache_data
def get_fvalue(val):
    feature_dict = {"No": 1, "Yes": 2}
    return feature_dict[val]

def get_value(val, my_dict): 
    return my_dict[val]


# Define app_mode using a Streamlit selectbox
app_mode = st.sidebar.selectbox('Choose the app mode', ['Home', 'Predict', 'About'])
if app_mode == 'Home': 
    st.title('Loan Prediction')
    st.image('chart_t1.png')
    st.markdown('Dataset:') 
    data = pd.read_csv('loan_data.csv')
    st.write(data.head())
    #st.bar_chart(data[['income_annum', 'loan_amount']].head(20))
    st.bar_chart(data[['income_annum', 'loan_amount']].head(20))
    print(data.columns) # to see the columns in the terminal
    st.write(data.columns) #to see the columns in the browsers

