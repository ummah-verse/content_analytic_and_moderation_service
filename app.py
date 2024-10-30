from flask import Flask, request, jsonify
import numpy as np
import io
import cv2
import uuid
import os
from PIL import Image, ImageOps
from transformers import pipeline, ViTFeatureExtractor, ViTForImageClassification, AutoModelForImageClassification, AutoProcessor, AutoModelForAudioClassification
from nudenet import NudeDetector
import tempfile
import torchaudio
import subprocess
import librosa
from moviepy.editor import VideoFileClip


from src.services.data_analysis import perform_data_analysis, fetch_data
from src.services.image_validator import validate_image
from src.services.video_validator import extract_frames
from src.middlewares.auth_middleware import token_required

app = Flask(__name__)

gender_model_name = "rizvandwiki/gender-classification"
gender_processor = AutoProcessor.from_pretrained(gender_model_name)
gender_classifier = AutoModelForImageClassification.from_pretrained(gender_model_name)

detector = NudeDetector()

age_classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")

    
allowed_score_threshold_covered = 0.2379505145549774
allowed_classes = [
    "FACE_FEMALE",
    "FACE_MALE",
]
covered_classes = [
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ANUS_COVERED",
]


# Endpoint to detect nudity in a single uploaded image
@app.route('/v3/detect-image-test', methods=['POST'])
def detect_image_test():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']

        # Simpan sementara untuk deteksi nudity
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file_path = tmp_file.name
            image_file.save(tmp_file_path)

        detections = detector.detect(tmp_file_path)
        
        print(detections)
        
        if not detections:
            os.remove(tmp_file_path)
            return jsonify({
                'message': 'success',
                # 'detections': detections,
                'gender': None,
                'age_group': None
            }), 200        
            
        os.remove(tmp_file_path)

        allowed = True
        gender = None
        for detection in detections:
            class_name = detection['class']
            score = detection['score']

            # khusus untuk feet covered 0.4 thresholdnya
            if class_name in covered_classes:
                if score < allowed_score_threshold_covered or (class_name == "FEET_COVERED" and score < 0.4):
                    continue
            if class_name in allowed_classes:
                gender = "female" if class_name == "FACE_FEMALE" else "male"
                continue  

            allowed = False
            break

        if allowed:
            # Perform gender classification
            image_pil = Image.open(image_file)
            gender_inputs = gender_processor(images=image_pil, return_tensors="pt")
            gender_outputs = gender_classifier(**gender_inputs)
            
            print(gender_outputs)
            
            gender_category = gender_outputs.logits.argmax(-1).item()  
            
            print(gender_category)
            
            gender = "female" if gender_category == 0 else "male"

            # Perform age classification
            image_pil_age = Image.open(image_file)
            age_prediction = age_classifier(image_pil_age)
            
            print(age_prediction)
            
            age_category = age_prediction[0]['label']
            
            print(age_category)
            
            if age_category in ["0-2"]:
                age_group = "baby"
            elif age_category in ["3-9"]:
                age_group = "children"
            elif age_category in ["10-19"]:
                age_group = "teen"
            else:
                age_group = "adult"     
            
            return jsonify({
                'message': 'success',
                # 'detections': detections,
                'gender': gender,
                'age_group': age_group
            }), 200
        else:
            return jsonify({'error': 'image is not valid'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Endpoint to detect nudity in a single uploaded image
@app.route('/v3/detect-image', methods=['POST'])
def detect_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file_path = tmp_file.name
            image_file.save(tmp_file_path)

        detections = detector.detect(tmp_file_path)

        os.remove(tmp_file_path)

        allowed = True
        for detection in detections:
            class_name = detection['class']
            score = detection['score']

            #khusus untuk feet covered 0.4 thresholdnya
            if class_name in covered_classes:
                if score < allowed_score_threshold_covered:
                    continue

            if class_name in allowed_classes:
                continue  

            allowed = False
            break

        if allowed:
            # cek jika face_male atau female_face
            # masukan ke json gender :  gender
            # jika tidak ada gender atau detections [], maka gender : null dan langsung response ke return jsonify
            # jika ada face_female atau male, maka gunakan foto sebelumnya untuk mengecek umurnya
            # deteksi foto apakah anak kecil atau bukan
            # Jika terlihat seperti umur 5 - 15, maka kategorikan sebagai child
            # jika terlihat seperti umur diatasnya, kategorikan sebagai adult
            # jika terlihat seperti umur 0 - 4 kategorikan sebagai baby
            # berikan success
            
            return jsonify({'message': 'success', 'detections': detections}), 200
        else:
            return jsonify({'error': 'image is not valid', 'detections' : detections}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


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

        # Daftar kelas yang diperbolehkan
        allowed_classes = ['Random', 'Female', 'Male', 'Hijab', 'Muscle']

        # Cek apakah class_name termasuk kelas yang diperbolehkan
        if class_name in allowed_classes:
            return jsonify({'message': 'success', 'type': class_name}), 200
        else:
            return jsonify({'error': 'image is not valid'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

from moviepy.editor import VideoFileClip

# Initialize the audio classification pipeline
# Initialize the audio classification pipeline
pipe = pipeline("audio-classification", model="MarekCech/GenreVim-Music-Detection-DistilHuBERT")

allowed_score_threshold_covered_video = 0.7379505145549774
allowed_classes_video = ["FACE_FEMALE", "FACE_MALE"]
covered_classes_video = ["FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED", "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ANUS_COVERED"]

@app.route('/predict/video', methods=['POST'])
def predict_video():
    try:
        if not request.data:
            return jsonify({'error': 'No video data received'}), 400

        video_dir = './temp_videos'
        os.makedirs(video_dir, exist_ok=True)

        video_id = str(uuid.uuid4())
        video_path = f'{video_dir}/{video_id}.mp4'

        with open(video_path, 'wb') as f:
            f.write(request.data)

        frames = extract_frames(video_path, interval=1)  # Extract frames based on your requirements
        
        nudity_detected = False
        valid_frames = 0
        gender = None

        for frame in frames:
            _, buffer = cv2.imencode('.jpg', frame)
            image_data = io.BytesIO(buffer).getvalue()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file_path = tmp_file.name
                tmp_file.write(image_data)

            detections = detector.detect(tmp_file_path)  # Define your detector
            os.remove(tmp_file_path)

            allowed = True
            for detection in detections:
                class_name = detection['class']
                score = detection['score']

                if class_name in covered_classes_video:
                    if score < allowed_score_threshold_covered_video or (class_name == "FEET_COVERED" and score < 0.4):
                        continue
                if class_name in allowed_classes_video:
                    gender = "female" if class_name == "FACE_FEMALE" else "male"
                    continue

                allowed = False

            if not allowed:
                nudity_detected = True
                break

            valid_frames += 1

        if nudity_detected:
            return jsonify({'error': 'Nudity detected in video frames'}), 400

        # Convert video to audio if no nudity was found
        audio_path = convert_video_to_audio(video_path)

        # Detect if audio contains music
        is_music = detect_music_in_audio(audio_path)
        os.remove(audio_path)  # Clean up audio file
        os.remove(video_path)

        print("yay")
        return jsonify({'message': 'success', 'valid_frames': valid_frames, 'gender': gender, 'music': is_music}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_video_to_audio(video_path):
    """
    Convert the first 10 seconds of a video file to audio (wav format).
    """
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    # Use moviepy to extract audio
    video_clip = VideoFileClip(video_path)
    
    # Get the first 10 seconds of the video
    audio_clip = video_clip.audio.subclip(0, min(10, video_clip.duration))
    audio_clip.write_audiofile(audio_path, codec='pcm_s16le')
    audio_clip.close()
    video_clip.close()
    return audio_path

def detect_music_in_audio(audio_path):
    """
    Detect if the audio contains music using librosa.
    """
    # Load the audio file and resample if necessary
    audio_input, sample_rate = librosa.load(audio_path, sr=None)  # Load at original sampling rate
    # Optionally, resample the audio
    target_sample_rate = 22050  # Adjust based on your model's requirement
    if sample_rate != target_sample_rate:
        audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=target_sample_rate)

    # Run the detection model
    result = pipe(audio_input)
    
    print(result)

    # Assuming class 'Music' is the target class for music detection
    return any(item['label'] == 'Music' and item['score'] > 0.5 for item in result)





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
    
    if any((result['label'] == 'toxicity' and result['score'] > 0.4) or 
        (result['label'] == 'black' and result['score'] > 0.8) or
        (result['label'] == 'sexual_explicit' and result['score'] > 0.3) or
        (result['label'] == 'threat' and result['score'] > 0.5)
        for result in results):
        print(results)
        return jsonify({"message": "content violates our policy"}), 400


    return jsonify({"message": "success"}), 200

@app.route('/check-text/reminder', methods=['POST'])
def check_text_reminder():
    data = request.get_json()

    # Extract the three text fields
    fields = {
        "text": data.get("text", ""),
        "title": data.get("title", ""),
        "location": data.get("location", "")
    }

    # Loop through each field and check for violations
    for field_name, content in fields.items():
        results = toxicity_model(content)
        if any(
            (result['label'] == 'toxicity' and result['score'] > 0.4) or
            (result['label'] == 'black' and result['score'] > 0.8) or
            (result['label'] == 'sexual_explicit' and result['score'] > 0.3) or
            (result['label'] == 'threat' and result['score'] > 0.5)
            for result in results
        ):
            print(f"Violation detected in {field_name}: {results}")
            return jsonify({"message": "content violates our policy"}), 400

    return jsonify({"message": "success"}), 200

@app.route('/check-text/yapping', methods=['POST'])
def check_text_yapping():
    data = request.get_json()

    # Extract the three text fields
    fields = {
        "text": data.get("text", ""),
        "location": data.get("location", "")
    }

    # Loop through each field and check for violations
    for field_name, content in fields.items():
        results = toxicity_model(content)
        if any(
            (result['label'] == 'toxicity' and result['score'] > 0.4) or
            (result['label'] == 'black' and result['score'] > 0.8) or
            (result['label'] == 'sexual_explicit' and result['score'] > 0.3) or
            (result['label'] == 'threat' and result['score'] > 0.5)
            for result in results
        ):
            print(f"Violation detected in {field_name}: {results}")
            return jsonify({"message": "content violates our policy"}), 400

    return jsonify({"message": "success"}), 200


@app.route('/check-text/profile', methods=['POST'])
def check_text_profile():
    data = request.get_json()

    # Optional fields
    fields = {
        "username": data.get("username"),
        "name": data.get("name"),
        "bio": data.get("bio")
    }

    # Check if at least one field is provided
    if all(value is None for value in fields.values()):
        return jsonify({"message": "success"}), 200

    # Loop through each provided field and run toxicity check
    for field_name, content in fields.items():
        if content:  # Only check if field is not None
            results = toxicity_model(content)
            if any(
                (result['label'] == 'toxicity' and result['score'] > 0.4) or
                (result['label'] == 'black' and result['score'] > 0.8) or
                (result['label'] == 'sexual_explicit' and result['score'] > 0.3) or
                (result['label'] == 'threat' and result['score'] > 0.5)
                for result in results
            ):
                print(f"Violation detected in {field_name}: {results}")
                return jsonify({"message": "content violates our policy"}), 400

    return jsonify({"message": "success"}), 200

    
if __name__ == '__main__':
    app.run(debug=True)
