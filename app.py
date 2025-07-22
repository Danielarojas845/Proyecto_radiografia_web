from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# === Cargar modelos en formato .keras ===
model_1 = load_model("modelo_afinado_tuning_v2.keras", compile=False)
model_2 = load_model("modelo_2_ensamble_v2.keras", compile=False)

# === Función de ensamble ===
def predict_ensemble(model1, model2, image_array, threshold=0.5):
    pred1 = model1.predict(image_array)[0][0]
    pred2 = model2.predict(image_array)[0][0]
    avg_pred = (pred1 + pred2) / 2
    return np.array([1 if avg_pred > threshold else 0]), avg_pred

# === Ruta principal ===
@app.route('/')
def index():
    return render_template('index.html')  # Asegúrate de tenerlo en /templates

# === Ruta de predicción ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen.'}), 400

    file = request.files['file']
    image_bytes = file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        image = image.resize((256, 256))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction, probabilidad = predict_ensemble(model_1, model_2, img_array)
        class_label = 'PNEUMONIA' if prediction[0] == 1 else 'NORMAL'

        return jsonify({
            'predicted_class': class_label,
            'probability': round(float(probabilidad), 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Ejecutar app ===
if __name__ == '__main__':
    app.run(debug=True, port=8000)












