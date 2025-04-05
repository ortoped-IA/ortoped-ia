import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Carrega o modelo treinado
MODEL_PATH = "classificador_ortopedia_ia.h5"
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['baixo_risco', 'moderado_risco', 'alto_risco']
IMG_SIZE = (224, 224)

def classify_image(image, dor, fratura, funcao_neurologica):
    image = image.convert('RGB')
    image = image.resize(IMG_SIZE)
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Ajuste clínico
    if dor == "Dor funcional":
        predicted_class = "alto_risco"
    elif predicted_class == "moderado_risco":
        if fratura == "Sim" or funcao_neurologica == "Déficit presente":
            predicted_class = "alto_risco"
    elif predicted_class == "baixo_risco":
        if dor != "Assintomático" or fratura == "Sim":
            predicted_class = "moderado_risco"

    return predicted_class, confidence

st.title("Classificador IA - Ortopedia Oncológica")
st.markdown("Envie uma radiografia e preencha os dados clínicos para classificar o risco de fragilidade e priorização.")

uploaded_file = st.file_uploader("Imagem de radiografia", type=["jpg", "jpeg", "png"])

dor = st.selectbox("Tipo de dor", ["Assintomático", "Dor leve", "Dor funcional"])
fratura = st.selectbox("Fratura patológica presente?", ["Não", "Sim"])
funcao_neurologica = st.selectbox("Função neurológica", ["Normal", "Déficit presente"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagem enviada", width=300)

    if st.button("Classificar"):
        predicted_class, confidence = classify_image(image, dor, fratura, funcao_neurologica)
        st.success(f"**Classificação: {predicted_class}** ({confidence:.2f}% de confiança)")
