from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
from PIL import Image
import io

from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

# Load the saved model in TensorFlow Lite format
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
@cross_origin()
def index():
    return "ML Model is working Fine"

# Route for image prediction
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    print('Predict route accessed')
    # Check if the 'image' file is present in the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400

    # Get the image file from the request
    image_file = request.files['image']

    # Load and preprocess the user-provided image
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data))
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    #input_details = interpreter.get_input_details()
    #output_details = interpreter.get_output_details()

    #interpreter.set_tensor(input_details[0]['index'], np.array([user_image]))
    #interpreter.invoke()
    # Preprocess the input image for TensorFlow Lite model
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_data = np.array(image, dtype=np.float32)
    # input_data = (input_data - 127.5) / 127.5  # Normalize image between -1 and 1
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference with the TensorFlow Lite model
    interpreter.invoke()

    # Get the output details and scale the predicted prices back to original values
    output_details = interpreter.get_output_details()
    predicted_normalized_prices = interpreter.get_tensor(output_details[0]['index'])
    predicted_prices = scaler.inverse_transform(predicted_normalized_prices)
    rounded_prices = np.round(predicted_prices, -3)
    rounded_prices = np.abs(rounded_prices)  # Apply absolute value

    predicted_price = float(rounded_prices[0][0])  # Convert to float

    return jsonify({'predicted_price': predicted_price}), 200

if __name__ == '__main__':
    app.run(debug=True)
