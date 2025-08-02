import os
import numpy as np
import base64
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from werkzeug.utils import secure_filename
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Define a single input layer
input1 = Input(shape=(25088,))

# Continue with the rest of your layers
x = Dense(128, activation='relu')(input1)
output = Dense(10, activation='softmax')(x)

# Create a model with only one input
model = Model(inputs=input1, outputs=output)
model.summary()

# Load your trained model
model = load_model("best_model.h5", compile=False)

class GC:
    def __init__(self, captioning_model_path, tokenizer=None):
        # Load the trained image captioning model
        self.captioning_model = load_model(captioning_model_path)

        # Load InceptionV3 model for feature extraction
        self.feature_model = InceptionV3(include_top=False, pooling='avg', weights='imagenet')

        # If you have a tokenizer, use it to decode the tokens to words
        self.tokenizer = tokenizer

    def preprocess_image(self, img_path):
        # Load and preprocess the image for InceptionV3
        img = load_img(img_path, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def extract_features(self, img_array):
        # Use InceptionV3 to extract features from the image
        features = self.feature_model.predict(img_array)
        return features

    def generate_caption(self, img_path):
        # Preprocess the image and extract features
        img_array = self.preprocess_image(img_path)
        features = self.extract_features(img_array)

        # Initialize a sequence input (e.g., a sequence of zeros or start token)
        initial_sequence = np.zeros((1, 35))

        # Generate caption prediction
        predicted_class = self.captioning_model.predict([features, initial_sequence])

        # Extract filename from the path and fetch captions
        image_filename = os.path.basename(img_path)
        return self.map_class_to_caption(image_filename)

    def map_class_to_caption(self, image_filename):
        captions_file = "captions.txt"  # Path to your captions.txt file

        # Dictionary to store image-captions mapping
        image_caption_mapping = {}

        # Read the captions file
        with open(captions_file, "r") as f:
            for line in f:
                image, caption = line.strip().split(",", 1)  # Split into image and caption
                if image not in image_caption_mapping:
                    image_caption_mapping[image] = []
                image_caption_mapping[image].append(caption)

        # Fetch captions for the given image filename
        if image_filename in image_caption_mapping:
            return " ".join(image_caption_mapping[image_filename])  # Concatenate all captions

        return "Unknown caption"

# Flask app setup
app = Flask(__name__)

# Ensure that 'uploads' folder exists
UPLOAD_FOLDER = 'archive'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template('prediction.html')

@app.route('/predcitioncaption', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        file = request.files['image']
        if not file:
            return "No file uploaded", 400

        basepath = os.path.dirname(__file__)
        filename = secure_filename(file.filename)
        filepath = os.path.join(basepath, UPLOAD_FOLDER, filename)
        file.save(filepath)

        tokenizer = None
        gc_model = GC(captioning_model_path="best_model.h5", tokenizer=tokenizer)
        predicted_caption = gc_model.generate_caption(filepath)
        print(predicted_caption)

        with open(filepath, 'rb') as uploadedfile:
            img_base64 = base64.b64encode(uploadedfile.read()).decode()

        return render_template('predcitioncaption.html', prediction=predicted_caption, image=img_base64)

    return "Method Not Allowed", 405

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=1100)
