import streamlit as st
import pickle
import numpy as np

# LOAD MODEL

with open("models/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# PAGE CONFIG

st.set_page_config(page_title="Iris Flower Prediction", layout="centered")

st.title("Iris Flower Classification")

st.write("Random Forest Classifier")

# SHOW HYPERPARAMETERS

st.subheader("Model Hyperparameters")

st.write("Number of Trees: 100")

st.write("Max Depth: 5")

st.write("Min Samples Split: 5")

st.write("Min Samples Leaf: 2")

# USER INPUTS

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)

sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)

petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)

petal_width = st.slider("Petal Width", 0.1, 3.0, 1.0)

# PREDICTION BUTTON

if st.button("Predict Flower"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    prediction = model.predict(input_data)

    flower_names = ["Setosa", "Versicolor", "Virginica"]

    result = flower_names[prediction[0]]

    st.success(f"Predicted Flower: {result}")
