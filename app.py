from flask import Flask, request, jsonify
import os
import tempfile
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('best.pt')  # Update with your model path


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Use a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        file_path = temp_file.name
        file.save(file_path)

    # Run inference
    results = model.predict(file_path)

    predictions = []

    for result in results:
        if result.probs is not None:
            # Get the top prediction class and its confidence
            top_class = result.probs.top1  # Assuming this is the index of the top class
            top_confidence = result.probs.top1conf  # Assuming this is the confidence of the top class
            predictions.append({
                'class': int(top_class),  # Convert to int if necessary
                'confidence': float(top_confidence)  # Convert to float if necessary
            })

    # Clean up the temporary file
    os.remove(file_path)

    # Return the predictions
    return jsonify({'predictions': predictions})


if __name__ == '__main__':
    app.run(debug=True)
