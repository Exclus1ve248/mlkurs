import os
import io
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, render_template
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'image'
app.config['PROCESSED_IMAGE_FOLDER'] = 'processed_images'

model = load_model('nukutCNN.h5')
model1 = load_model('nikit_model.h5')

model2 = YOLO('road.pt')

classes = {
    0: 'speed_limit',
    1: 'stop',
    2: 'traffic_light',
    3: 'zebra'
}
classes2 = {
    0: 'speed_limit',
    1: 'stop',
    2: 'traffic_light',
    3: 'zebra'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/original_image/<filename>')
def original_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

@app.route('/processed_image/<filename>')
def processed_image(filename):
    return send_from_directory(app.config['PROCESSED_IMAGE_FOLDER'], filename, as_attachment=False)

@app.route('/api_traffic_sign', methods=['POST'])

# curl -X POST -H "Content-Type: multipart/form-data" -F "image=@C:\Users\nicki\Desktop\image.jpg" http://localhost:5000/api_traffic_sign

def predict_traffic_sign_api():
    if 'image' not in request.files:
        return jsonify({"error": "Bad request. No image part."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Bad request. No selected file."}), 400

    img = Image.open(file.stream)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model1.predict(img_array)  # Use model1 here
    class_idx = np.argmax(predictions)

    class_name = classes[class_idx]

    result = {
        "class": class_name,
        "probability": predictions[0, class_idx].item()
    }

    return jsonify(result), 200

@app.route('/predict1', methods=['POST'])
def predict1():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model.predict(x)
    class_idx = np.argmax(pred)

    class_name = classes[class_idx]

    result = f'{class_name} ({pred[0, class_idx].item():.4f})'
    return result

@app.route('/predict2', methods=['POST'])
def predict2():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    img = image.load_img(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model1.predict(x)
    class_idx = np.argmax(pred)

    class_name = classes[class_idx]

    result = f'{class_name} ({pred[0, class_idx].item():.4f})'
    return result

@app.route('/predict3', methods=['POST'])
def predict3():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    results = model2(img)[0]
    boxes = results.boxes
    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()

    original_filename = file.filename
    processed_filename = f"processed_{file.filename}"
    original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    processed_file_path = os.path.join(app.config['PROCESSED_IMAGE_FOLDER'], processed_filename)

    results_plot = results.plot()
    results_image = Image.fromarray(results_plot)
    results_image.save(processed_file_path)

    return render_template('result.html', original_filename=original_filename, processed_filename=processed_filename)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['PROCESSED_IMAGE_FOLDER']):
        os.makedirs(app.config['PROCESSED_IMAGE_FOLDER'])
    app.run(debug=True)