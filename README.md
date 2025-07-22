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

- `modelo_afinado_tuning_v2.keras': : CNN ajustada con tuning de hiperparámetros  
- `modelo_2_ensamble_v2.keras': Modelo estandar, para hacer ensamble.

📌 **Estrategia de ensamble**: Promedio simple de ambas salidas para robustecer el diagnóstico.

⚠️ Los modelos no están almacenados en GitHub debido a su tamaño. Se alojan en Google Drive y se pueden descargar directamente desde el entorno de ejecución.

---

## 📦 Descarga y carga de modelos entrenados

### 🔽 Descarga automática desde Google Drive
 
```python
!pip install gdown

import gdown

# Descargar ZIPs correctamente
gdown.download(id='1lQL22NS13Bn-3mXAr-U1h_VGTp7V9gR-', output='modelo_2_ensamble.zip', quiet=False)
gdown.download(id='1O7IQhH-nnozax5iQDVpih-SkRDLAUh0O', output='modelo_afinado_tuning.zip', quiet=False)

import zipfile

with zipfile.ZipFile('modelo_2_ensamble.zip', 'r') as zip_ref:
    zip_ref.extractall()

with zipfile.ZipFile('modelo_afinado_tuning.zip', 'r') as zip_ref:
    zip_ref.extractall()

from tensorflow.keras.models import load_model

modelo1 = load_model('modelo_2_ensamble.keras')
modelo2 = load_model('modelo_afinado_tuning.keras')

from tensorflow.keras.models import load_model
from google.colab import drive

# Montar Google Drive
drive.mount('/content/drive')

# === Paso 1: Cargar los modelos originales (están en /content)
modelo1 = load_model('/content/modelo_afinado_tuning.keras', compile=False)
modelo2 = load_model('/content/modelo_2_ensamble.keras', compile=False)

# === Paso 2: Guardar en Google Drive con nuevo nombre
modelo1.save('/content/drive/MyDrive/modelo_afinado_tuning_v2.keras', save_format='keras')
modelo2.save('/content/drive/MyDrive/modelo_2_ensamble_v2.keras', save_format='keras')

print(" ¡Modelos guardados en Google Drive!")

# === Paso 3: Copiar desde Google Drive a Colab (si los quieres volver a usar desde /content)
!cp "/content/drive/MyDrive/modelo_afinado_tuning_v2.keras" /content/
!cp "/content/drive/MyDrive/modelo_2_ensamble_v2.keras" /content/

# === Paso 4: Cargar los modelos guardados (versión v2)
modelo1 = load_model('/content/modelo_afinado_tuning_v2.keras', compile=False)
modelo2 = load_model('/content/modelo_2_ensamble_v2.keras', compile=False)

print(" ¡Modelos cargados correctamente desde Google Drive!")

# === IMPORTAR LIBRERÍAS NECESARIAS ===
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# === CARGAR LOS MODELOS CORRECTOS ===
modelo1 = load_model("modelo_afinado_tuning_v2.keras", compile=False)
modelo2 = load_model("modelo_2_ensamble_v2.keras", compile=False)

# === FUNCIÓN DE ENSAMBLE CON PROMEDIO DE PROBABILIDADES ===
def predict_ensemble(model1, model2, image_array, threshold=0.5):
    pred1 = model1.predict(image_array)[0][0]
    pred2 = model2.predict(image_array)[0][0]
    avg_pred = (pred1 + pred2) / 2
    class_label = 'PNEUMONIA' if avg_pred > threshold else 'NORMAL'
    return class_label, avg_pred

# === CARGAR Y PREPROCESAR IMAGEN DE PRUEBA ===
image_path = "/content/003-radiografia-neumonia.jpg.webp"  # Cambia la ruta si es necesario
image = Image.open(image_path).convert("L")
image = image.resize((256, 256))
img_array = img_to_array(image) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === HACER PREDICCIÓN ===
class_label, prob = predict_ensemble(modelo1, modelo2, img_array)

# === MOSTRAR RESULTADO Y VISUALIZACIÓN ===
print(f"Diagnóstico: {class_label} ({prob * 100:.2f}% de certeza)")
plt.imshow(image, cmap='gray')
plt.title(f"Predicción: {class_label} ({prob * 100:.2f}%)")
plt.axis('off')
plt.show()



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

👉 [Abrir en Google Colab](https://colab.research.google.com/drive/1Jsp4F5OPBDt0yC_bq4o78nfmS0GgV1ot#scrollTo=rIb41jhG-K9r&uniqifier=2)

👩‍💼 Autora
Daniela Rojas
Enfermera e Ingeniera Comercial | Data Scientist | Con interés por la salud digital y la inteligencia artificial aplicada a medicina.

📃 Licencia
Este proyecto se distribuye bajo la licencia Credly by Pearson.
