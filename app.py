# app.py

import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import numpy as np
import base64
import io

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('emnist_cnn_62_class_v1.h5')
print("Model loaded.")

label_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def preprocess_image_from_web(image_data):
    img_str = image_data.split(',')[1]
    img_bytes = base64.b64decode(img_str)
    
    img = Image.open(io.BytesIO(img_bytes))

    img_with_bg = Image.new("RGB", img.size, "WHITE")
    img_with_bg.paste(img, (0, 0), img)

    img_L = img_with_bg.convert('L')

    # Flip image (like EMNIST dataset)
    img_L = img_L.transpose(Image.Transpose.TRANSPOSE)

    # Invert colors (input is black on white background)
    img_L = ImageOps.invert(img_L)

    # Resize to the model's required 28x28 input size
    img_L = img_L.resize((28, 28))

    # Convert to NumPy array and normalize
    img_array = np.array(img_L) / 255.0

    # Reshape for the model: (batch_size, height, width, channels) -> (1, 28, 28, 1)
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['imageData']

    processed_image = preprocess_image_from_web(image_data)
    
    if processed_image is not None:
        prediction = model.predict(processed_image)

        predicted_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        predicted_character = label_mapping[predicted_index]
        
        return jsonify({
            'prediction': predicted_character,
            'confidence': confidence
        })
    else:
        return jsonify({'error': 'Could not process image'}), 400

if __name__ == '__main__':
    app.run(debug=True)