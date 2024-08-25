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

def validate_image(file):
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image).astype(np.float32)
        normalized_image = (image_array / 127.5) - 1
        data = np.expand_dims(normalized_image, axis=0)

        infer = model.signatures['serving_default']
        input_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
        predictions = infer(input_tensor)
        output = predictions['sequential_3']  
        output_array = output.numpy()
        class_index = np.argmax(output_array)

        class_names = ["Anjing", "Cewek Normal", "Cewek Hijab - Abaya", "Cewe Gabener", "Cowo Jelek", "Random"]  
        class_name = class_names[class_index]

        return class_name

    except Exception as e:
        print(f"Error: {e}")
        return None
