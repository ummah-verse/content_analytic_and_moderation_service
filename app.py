from flask import Flask, request, jsonify
import numpy as np
import io
import cv2
import uuid
import os
from transformers import pipeline

from src.services.data_analysis import perform_data_analysis, fetch_data
from src.services.image_validator import validate_image
from src.services.video_validator import extract_frames
from src.middlewares.auth_middleware import token_required

app = Flask(__name__)

@app.route('/predict/image', methods=['POST'])
def predict():
    try:
        image_data = request.data
        if not image_data:
            return jsonify({'error': 'No image data received'}), 400

        class_name = validate_image(image_data)
        
        print(class_name)

        if class_name is None:
            return jsonify({'error': 'Internal Server Error'}), 500

        if class_name == 'Cewek Hijab - Abaya':
            return jsonify({'message': 'success'}), 200
        elif class_name == 'Cewe Gabener':
            return jsonify({'message': 'image is not valid'}), 400
        elif class_name == 'Cewek Normal':
            return jsonify({'message': 'success'}), 200
        elif class_name == 'Cowo Jelek':
            return jsonify({'message': 'success'}), 200
        elif class_name == 'Random':
            return jsonify({'message': 'success'}), 200
        else:
            return jsonify({'message': 'success'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/predict/video', methods=['POST'])
def predict_video():
    try:
        if not request.data:
            return jsonify({'error': 'No video data received'}), 400

        video_dir = './temp_videos'
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        
        video_id = str(uuid.uuid4())
        video_path = f'{video_dir}/{video_id}.mp4'

        with open(video_path, 'wb') as f:
            f.write(request.data)

        frames = extract_frames(video_path, interval=1)
        
        valid_frames = 0
        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = io.BytesIO(buffer).getvalue()

            class_name = validate_image(image_data)
            if class_name == 'Cewek Hijab - Abaya':
                valid_frames += 1

        os.remove(video_path)

        if valid_frames > 0:
            return jsonify({'message': 'success', 'valid_frames': valid_frames}), 200
        else:
            return jsonify({'message': 'success'}), 400
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/data-analysis', methods=['GET'])
def data_analysis():
    """Route untuk menjalankan analisis data."""
    result = perform_data_analysis()
    return jsonify(result)

@app.route('/user', methods=['GET'])
def get_user_controller():
    result = fetch_data()
    return jsonify(result)

toxicity_model = pipeline("text-classification", model="unitary/unbiased-toxic-roberta")

@app.route('/check-text', methods=['POST'])
def check_text():
    data = request.get_json()
    text = data.get("text", "")

    results = toxicity_model(text)
    print(results)
    
    if any(result['label'] == 'toxicity' and result['score'] > 0.4 for result in results):
        print(results)
        return jsonify({"message": "content violate our policy"}), 400
    
    return jsonify({"message": "success"}), 200


    
if __name__ == '__main__':
    app.run(debug=True)
