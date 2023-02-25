from flask import Flask, request, jsonify
import os
import tensorflow as tf
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import random
from collections.abc import Mapping
from flask_cors import CORS
import string

app = Flask(__name__, static_url_path='/home/tadi_virinchi/static')
CORS(app)

OUTPUT_DIR = 'static/uploaded_images'

def generate_filename():
    return ''.join(random.choices(string.ascii_lowercase, k=20)) + '.jpg'

@app.route("/", methods=["POST"])
def process_image():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                absolute_path = os.path.abspath("../tadi_virinchi/"+OUTPUT_DIR+"/"+generate_filename())
                #app
                print(absolute_path)

                uploaded_file.save(absolute_path)
	
                img = load_img(absolute_path, grayscale=True)
                img = img.resize((64,64))
                img_array = img_to_array(img)
                img = img_array / 255.0
                img = np.expand_dims(img,0)

                reconstructed_model = tf.keras.models.load_model("/home/tadi_virinchi/model-ml")
                pred = reconstructed_model.predict(img)
                y_pred = [np.argmax(element) for element in pred]
                
                predicted_value = ""

                if y_pred[0] == 0:
                    predicted_value = "glioma"
                elif y_pred[0] == 1:
                    predicted_value = "meningioma"
                elif y_pred[0] == 2:
                    predicted_value = "no tumor"
                elif y_pred[0] == 3:
                    predicted_value = "pituitary"

                return jsonify({'msg': 'success', 'predicted_class': predicted_value, 'mri':absolute_path})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))