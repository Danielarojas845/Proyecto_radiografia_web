# 📊 Clasificador de Radiografías de Tórax con Deep Learning

Este proyecto implementa una red neuronal convolucional (CNN) usando Keras y Flask para detectar neumonía en radiografías de tórax. Se basa en un ensamble de dos modelos, lo que mejora la precisión y robustez del sistema.

---

## 🔧 Tecnologías utilizadas

- Python  
- TensorFlow / Keras
- flask-cors
- Flask (API)  
- NumPy, Pillow  
- Google Colab (entrenamiento)  
- GitHub (repositorio)

---

## 📂 Dataset

Se utilizó el dataset de Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

- Imágenes en escala de grises (256x256)
- Dataset balanceado:  
  - 1.072 imágenes NORMAL  
  - 1.073 imágenes PNEUMONIA

---

## 🎓 Modelos entrenados

Este proyecto incluye dos modelos entrenados con Keras para clasificación binaria:

- `modelo_afinado_tuning.keras`: CNN ajustada con tuning de hiperparámetros  
- `modelo_2_ensamble.keras`: Modelo estandar, para hacer ensamble.  

📌 **Estrategia de ensamble**: Promedio simple de ambas salidas para robustecer el diagnóstico.

⚠️ Los modelos no están almacenados en GitHub debido a su tamaño. Se alojan en Google Drive y se pueden descargar directamente desde el entorno de ejecución.

---

## 📦 Descarga y carga de modelos entrenados

### 🔽 Descarga automática desde Google Drive

```python
!pip install gdown
import gdown
import zipfile

# Descargar archivos .zip
gdown.download(id='1lQL22NS13Bn-3mXAr-U1h_VGTp7V9gR-', output='modelo_2_ensamble.zip', quiet=False)
gdown.download(id='1O7IQhH-nnozax5iQDVpih-SkRDLAUh0O', output='modelo_afinado_tuning.zip', quiet=False)

# Extraer
with zipfile.ZipFile('modelo_2_ensamble.zip', 'r') as zip_ref:
    zip_ref.extractall()

with zipfile.ZipFile('modelo_afinado_tuning.zip', 'r') as zip_ref:
    zip_ref.extractall()

🧠 Carga en memoria
from tensorflow.keras.models import load_model

modelo1 = load_model('modelo_2_ensamble.keras')
modelo2 = load_model('modelo_afinado_tuning.keras')

📥 Enlaces de descarga manual
modelo_2_ensamble.zip

modelo_afinado_tuning.zip

🌐 API con Flask
Ruta disponible:

http
Copiar
Editar
POST /predict
Envía una imagen en el parámetro file y recibe una respuesta como:

json
Copiar
Editar
{
  "predicted_class": "PNEUMONIA",
  "confidence": "94.27%"
}
📊 Métricas del modelo ensamblado
Precisión (Accuracy) en test: 73%

AUC ROC: 0.94

Recall para clase PNEUMONIA: 0.99

## 🧪 Ejecutar en Google Colab

Puedes probar este proyecto directamente desde Google Colab:

👉 [Abrir en Google Colab]()

👩‍💼 Autora
Daniela Rojas
Enfermera e Ingeniera Comercial | Data Scientist | Con interés por la salud digital y la inteligencia artificial aplicada a medicina.

📃 Licencia
Este proyecto se distribuye bajo la licencia Credly by Pearson.
