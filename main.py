import streamlit as st
import pickle
import numpy as np

# Set basic page config
st.title('Iris Flower Prediction')
st.write('Enter the measurements below to predict the Iris flower type')

# Load the saved model and scaler
try:
    with open('model\iris_knn_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('model\iris_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Create input fields
    sepal_length = st.number_input('Sepal Length (cm)', value=5.0)
    sepal_width = st.number_input('Sepal Width (cm)', value=3.5)
    petal_length = st.number_input('Petal Length (cm)', value=1.4)
    petal_width = st.number_input('Petal Width (cm)', value=0.2)

    # Create a button for prediction
    if st.button('Predict'):
        # Prepare input data
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        # Map prediction to iris type
        iris_types = ['Setosa', 'Versicolor', 'Virginica']
        predicted_type = iris_types[prediction[0]]
        
        # Show prediction
        st.success(f'Predicted Iris Type: {predicted_type}')

except FileNotFoundError:
    st.error("Model files not found. Please ensure 'iris_knn_model.pkl' and 'iris_scaler.pkl' are in the same directory.")