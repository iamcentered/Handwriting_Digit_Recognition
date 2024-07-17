from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)
model = tf.keras.models.load_model('digit_classifier.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = Image.open(BytesIO(image_data)).convert('L').resize((28, 28))
    image = np.array(image).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)
    return jsonify({'digit': int(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)