from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, '..', 'model', 'yappin', 'savedmodel')

if not tf.io.gfile.exists(model_dir):
    raise Exception(f"Model directory not found: {model_dir}")

# Muat model TensorFlow
model = tf.saved_model.load(model_dir)
print("Model loaded successfully.")

def validate_image(image_data):
    try:
        # Convert the raw image buffer (received from Node.js) into a PIL Image object
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Resize and preprocess the image
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32)

        # Normalize the image (standard TensorFlow normalization)
        normalized_image = (image_array / 127.5) - 1

        # Add batch dimension
        data = np.expand_dims(normalized_image, axis=0)

        # Perform inference using TensorFlow model
        infer = model.signatures['serving_default']
        input_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
        predictions = infer(input_tensor)

        # Assuming the output layer is called 'sequential_3', adjust if needed
        output = predictions['sequential_3']  
        output_array = output.numpy()

        # Get the class index with the highest confidence
        class_index = np.argmax(output_array)

        # Class names for the model
        class_names = ["Anjing", "Cewek Normal", "Cewek Hijab - Abaya", "Cewe Gabener", "Cowo Jelek", "Random"]
        class_name = class_names[class_index]
        
        print(class_name)

        return class_name

    except Exception as e:
        print(f"Error: {e}")
        return None
    
# def validate_image(file):
#     try:
#         image = Image.open(io.BytesIO(file.read())).convert("RGB")
#         image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
#         image_array = np.asarray(image).astype(np.float32)
#         normalized_image = (image_array / 127.5) - 1
#         data = np.expand_dims(normalized_image, axis=0)

#         infer = model.signatures['serving_default']
#         input_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
#         predictions = infer(input_tensor)
#         output = predictions['sequential_3']  
#         output_array = output.numpy()
#         class_index = np.argmax(output_array)

#         class_names = ["Anjing", "Cewek Normal", "Cewek Hijab - Abaya", "Cewe Gabener", "Cowo Jelek", "Random"]  
#         class_name = class_names[class_index]

#         return class_name

#     except Exception as e:
#         print(f"Error: {e}")
#         return None
