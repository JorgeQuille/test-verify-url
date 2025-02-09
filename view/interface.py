import streamlit as st
import plotly.express as px
import pandas as pd
from model.UrlClassifier import UrlClassifier

def bar_chart(df):
    pred_count = df['prediccion'].value_counts().reset_index()
    pred_count.columns = ['prediccion', 'cantidad'] 
    fig = px.bar(pred_count, 
                    x='prediccion', 
                    y='cantidad', 
                    title='Cantidad de registros por prediccion',
                    labels={'cantidad': 'Cantidad de Registros', 'prediccion': 'predicción'}, 
                    color='prediccion',
                    text='cantidad'
                )

    st.plotly_chart(fig)

    pred_count['porcentaje'] = (pred_count['cantidad'] / pred_count['cantidad'].sum()) * 100
    #pred_count['porcentaje'] = pred_count['porcentaje'].map(lambda x: f'{x:.2f}%')
    
    fig1 = px.bar(pred_count, 
                    x='prediccion', 
                    y='porcentaje', 
                    title='Porcentaje de Registros por Predicción',
                    labels={'porcentaje': 'Porcentaje (%)', 'prediccion': 'Predicción'}, 
                    text='porcentaje'
                )

    fig1.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig1.update_yaxes(tickvals=list(range(0, 101, 10)), ticktext=[f'{i}%' for i in range(0, 101, 10)])
    st.plotly_chart(fig1)

def view_interface():
    urlClassifier = UrlClassifier()
    st.title("Clasificador de URL")
    url = st.text_input("Ingresa la URL:")
    st.write("O")
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
    
    if uploaded_file is not None:
        column_name = st.text_input("Ingresa el nombre de la columna a tratar:", "")
        df = pd.read_csv(uploaded_file) 
        if column_name:
            if column_name in df.columns:
                model_name = st.selectbox("Selecciona el modelo:", urlClassifier.get_models()) 
                if st.button("Predecir"):
                    df_result = urlClassifier.predict_csv(model_name, df, column_name)
                    bar_chart(df_result)
                    csv = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Descargar CSV con Predicciones",
                        data=csv,
                        file_name='predicciones.csv',
                        mime='text/csv',
                    )
            else:
                st.error(f"La columna '{column_name}' no se encuentra en los datos cargados.")
    elif url:
        model_name = st.selectbox("Selecciona el modelo:", urlClassifier.get_models()) 
        if st.button("Predecir"):
            prediction, probability = urlClassifier.predic(url, model_name)
            st.write(f"Predicción: {prediction}")
            st.write(f"Probabilidad: {probability:.2f}")