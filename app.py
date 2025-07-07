from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv8n model once at startup
model = YOLO('yolov8n.pt')

@app.route('/')
def index():
    return 'Food Calorie Estimator API – upload an image with POST /upload'


def run_detection(image_path: str):
    """Run YOLOv8n on the given image and return basic detection results."""
    results = model(image_path)
    detections = []
    names = model.names

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]
            detections.append({
                'label': label,
                'confidence': round(conf, 2)
            })
    return detections


@app.route('/upload', methods=['POST'])
def upload():
    # Validate request
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save image
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Run detection
    detections = run_detection(file_path)

    return jsonify({
        'filename': filename,
        'detections': detections
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)