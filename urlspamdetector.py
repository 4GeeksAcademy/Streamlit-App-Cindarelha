# your code here
import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

#import os
#print(os.listdir('.'))

# Cargar el modelo previamente entrenado (suponiendo que esté guardado como 'spam_model.pkl')
model = joblib.load("svm_spam_detector.pkl")

# Cargar el vectorizador
vectorizer = joblib.load("vectorizer.pkl")

# Título de la aplicación
st.title("Detección de URLs Spam")

# Descripción
st.write("Esta aplicación predice si una URL es spam o no spam utilizando un modelo de Machine Learning.")

# Entrada de usuario para URL
user_input = st.text_input("Ingrese una URL para analizar:")

# Procesamiento del texto
if user_input:
    # Transformar la entrada del usuario con el vectorizador
    user_input_vectorized = vectorizer.transform([user_input])

    # Realizar la predicción
    prediction = model.predict(user_input_vectorized)

    # Mostrar el resultado
    if prediction[0] == 1:
        st.error("Esta URL es Spam!")
    else:
        st.success("Esta URL no es Spam.")