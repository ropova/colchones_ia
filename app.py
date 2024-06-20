import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Configurar TensorFlow en modo compatibilidad (para evitar algunos warnings)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Cargar el modelo entrenado
model = load_model('modelo_colchones.h5')  # Asegúrate de tener el archivo del modelo correcto y la ruta adecuada

# Compilación del modelo
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Forzar la construcción de métricas
model.built = True

# Función para realizar la predicción
def predict(colchon_data):
    # Normalizar los datos de entrada (si se normalizaron durante el entrenamiento)
    # Aquí puedes aplicar la misma normalización que se usó durante el entrenamiento
    # Esto puede incluir convertir las características categóricas en códigos numéricos, o normalizar numéricamente
    
    # Asegurar que los datos de entrada tengan la forma correcta (6 características)
    if len(colchon_data) != 6:
        st.error('El número de características ingresadas no es válido.')
        return None

    # Realizar la predicción
    X_pred = np.array([colchon_data])  # Convertir a un arreglo numpy
    prediction = model.predict(X_pred)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Interfaz de la aplicación con Streamlit
def main():
    st.title('Predicción de Tipo de Colchón')

    # Preguntas para el usuario
    st.subheader('Por favor, selecciona las opciones de acuerdo a tus necesidades:')
    
    # Definir las opciones para cada característica
    firmeza_options = ['Suave', 'Media', 'Firme']
    tamaño_options = ['Individual', 'Matrimonial', 'Queen', 'King']
    posición_options = ['Lateral', 'Espalda', 'Estómago']
    presupuesto_options = ['Bajo', 'Medio', 'Alto']

    # Interacción del usuario
    firmeza = st.selectbox('Firmeza:', firmeza_options)
    tamaño = st.selectbox('Tamaño:', tamaño_options)
    posición = st.selectbox('Posición de dormir:', posición_options)
    presupuesto = st.selectbox('Presupuesto:', presupuesto_options)

    # Convertir las opciones seleccionadas a números
    firmeza_dict = {'Suave': 0, 'Media': 1, 'Firme': 2}
    tamaño_dict = {'Individual': 0, 'Matrimonial': 1, 'Queen': 2, 'King': 3}
    posición_dict = {'Lateral': 0, 'Espalda': 1, 'Estómago': 2}
    presupuesto_dict = {'Bajo': 0, 'Medio': 1, 'Alto': 2}

    # Recolectar los datos del colchón en un arreglo (debe tener 6 características)
    colchon_data = [
        firmeza_dict.get(firmeza, 0),
        tamaño_dict.get(tamaño, 0),
        posición_dict.get(posición, 0),
        presupuesto_dict.get(presupuesto, 0),
        firmeza_dict.get(firmeza, 0),
        presupuesto_dict.get(presupuesto, 0)
    ]

    # Verificar si se presiona el botón de predicción
    if st.button('Recomendar'):
        # Realizar la predicción
        prediction = predict(colchon_data)
        if prediction is not None:
            # Mostrar el resultado de la predicción
            tipos_colchones = ['Muelles Bicónicos', 'Látex', 'Muelles Ensacados', 'Espumación HR', 'Híbrido']
            st.success(f'El tipo de colchón recomendado es: {tipos_colchones[prediction]}')

            # Mostrar la imagen correspondiente
            imagenes_colchones = {
                0: 'static/images/muelles_biconicos.jpeg',
                1: 'static/images/latex.jpg',
                2: 'static/images/muelle_ensacado.jpg',
                3: 'static/images/espumacion_hr.jpg',
                4: 'static/images/hibrido.jpg'
            }
            
            image_path = imagenes_colchones.get(prediction)
            st.image(image_path, caption=tipos_colchones[prediction], use_column_width=True)

# Iniciar la aplicación
if __name__ == '__main__':
    main()
