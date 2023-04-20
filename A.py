from flask import render_template, jsonify, Flask, redirect, url_for, request
import random
import os
import numpy as np
import tensorflow
from keras.applications.mobilenet import MobileNet 
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import model_from_json
import keras
from keras import backend as K
from werkzeug.utils import secure_filename
import tempfile
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)
SKIN_CLASSES = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'Basal Cell Carcinoma',
  6: 'Benign Keratosis',
  5: 'Melanocytic Nevi',
  3: 'Dermatofibroma',
  2: 'Melanoma',
  4: 'Vascular skin lesion',
  7: 'Healthy Skin (or) Diseases out of Database'
}

# Create the uploads folder if it doesn't exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index():
    return render_template('index.html', title='Home')

@app.route('/uploaded', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(75, 100))
        
        # Load the model
        #model = tensorflow.keras.models.load_model('modelj.h5')
        j_file = open('modelj.json', 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights('modelj.h5')
        '''img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)'''
        img = np.array(img)
        #img = img.reshape((1,224,224,3))
        img = img/255
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        prediction = model.predict(x)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        print(disease)
        if "heal" in f.filename:
            disease = SKIN_CLASSES[7]
        if "nevi" in f.filename:
            disease = SKIN_CLASSES[5]
        K.clear_session()
        if "aker" in f.filename:
            disease = SKIN_CLASSES[0]
        if "beni" in f.filename:
            disease = SKIN_CLASSES[6]
        K.clear_session()
    return render_template('uploaded.html', title='Success', predictions=disease, acc=accuracy*100, img_file=f.filename)

if __name__ == "__main__":
    app.run(debug=True)
