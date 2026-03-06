import streamlit as st
import numpy as np 
from sklearn.externals import joblib

model = joblib.load("iris_model.pkl")
st.title("Iris Prediction")

sepal_len = st.number_input("Sepal Length")
sepal_wid = st.number_input("Sepal Width")
petal_len = st.number_input("Petal Length")
petal_wid = st.number_input("Petal Width")

if st.button("predict"):
    input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    prediction = model.predict(input_data)

    flowers = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"prediction: {flowers[prediction[0]]}")