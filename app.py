from flask import Flask, request, jsonify
from src.services.data_analysis import perform_data_analysis, fetch_data
from src.services.image_validator import validate_image
from src.middlewares.auth_middleware import token_required

app = Flask(__name__)

# @token_required
# @app.route('/predict', methods=['POST'])
# def predict(current_user):
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    class_name = validate_image(file)
    
    if class_name is None:
        return jsonify({'error': 'Internal Server Error'}), 500
    
    if class_name == 'Cewek Hijab - Abaya':
        return jsonify({'message': 'success'}), 200
    elif class_name == 'Anjing':
        return jsonify({'message': 'image is not valid'}), 400
    elif class_name == 'Cowo Jelek':
        return jsonify({'message': 'ini cowo jelek'}), 400
    else:
        return jsonify({'message': 'unknown class'}), 400

@app.route('/data-analysis', methods=['GET'])
def data_analysis():
    """Route untuk menjalankan analisis data."""
    result = perform_data_analysis()
    return jsonify(result)

@app.route('/user', methods=['GET'])
def get_user_controller():
    result = fetch_data()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
