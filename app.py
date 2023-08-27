# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 08:51:50 2023

@author: TayyubKhan
"""

from flask import Flask, request, jsonify, abort
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('emnist_model_digits.keras')

@app.route('/predict', methods=['POST'])
def predict_digit():
    try:
        # Check if 'image' is in request files
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided in the request.'}), 400

        image_data = request.files['image'].read()
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to model's input size
        image_array = np.array(image)
        input_image = np.expand_dims(image_array, axis=0)
        input_image = input_image / 255.0

        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction)

        return jsonify({
            'predicted_class': int(predicted_class)
        }), 200

    except KeyError:
        return jsonify({'error': 'No image provided in the request.'}), 400
    except FileNotFoundError:
        return jsonify({'error': 'Model file not found.'}), 500
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run()
