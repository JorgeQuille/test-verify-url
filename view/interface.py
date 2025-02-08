import streamlit as st
from model.UrlClassifier import UrlClassifier

def view_interface():
    urlClassifier = UrlClassifier()
    st.title("Clasificador de URL")
    url = st.text_input("Ingresa la URL:")
    model_name = st.selectbox("Selecciona el modelo:", urlClassifier.get_models()) 
    
    if st.button("Predecir"):
        if url:
            prediction, probability = urlClassifier.predic(url, model_name)
            st.write(f"Predicción: {prediction}")
            st.write(f"Probabilidad: {probability:.2f}")
        else:
            st.write('Por favor ingrese una URL')