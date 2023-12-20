from flask import Flask, render_template, request, jsonify, url_for
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.applications.inception_v3 import preprocess_input
import time
app = Flask(__name__)

# Load the trained model from the HDF5 file
model = tf.keras.models.load_model('model4.h5')
target_size = (299, 299)

def load_and_preprocess_new_image(file_path, target_size=target_size):
    img = load_img(file_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index2.html', uploaded_images=None, dirt_image_names=None)

@app.route('/app1')
def app1_page():
    return render_template('app1_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'folder' not in request.files:
        return jsonify({'error': 'No folder provided'})

    uploaded_folder = 'static/uploaded_folder'
    if not os.path.exists(uploaded_folder):
        os.makedirs(uploaded_folder)

    folder = request.files.getlist('folder')
    if not folder:
        return jsonify({'error': 'No selected folder'})
    
    # Iterate through the images in the folder
    uploaded_images = []
    dirt_image_names = []  # List to store names of all uploaded images
    for file_contents in folder:
    # Construct a unique filename for the current image
        filename = os.path.join(uploaded_folder, os.path.basename(file_contents.filename.replace("/", "\\")))

        # Save the image temporarily
        file_contents.save(filename)

        # Load and preprocess the new image
        new_image = load_and_preprocess_new_image(filename)
        # Measure the time taken for prediction
        start_time = time.time()
        # Make predictions using the trained model
        prediction_result = model.predict(np.expand_dims(new_image, axis=0))
        prediction_prob = float(prediction_result[0][0])
         # Calculate the time taken for prediction
        end_time = time.time()
        elapsed_time = end_time - start_time
        # Interpret the prediction result
        prediction_label = "Dirt Buildup" if prediction_prob > 0.5 else "Clean"
        # Store the result in a dictionary
        result = {
                'image': url_for('static', filename=f'uploaded_folder/{os.path.basename(file_contents.filename)}'),
                'file_path': filename,
                'prediction_label': prediction_label,
                'prediction_prob': prediction_prob,
                'elapsed_time': elapsed_time
                }
         # Only include dirt images in the result
        if prediction_label == "Dirt Buildup":

        # Append the result to the list of uploaded images
             uploaded_images.append(result)
             dirt_image_names.append(os.path.basename(file_contents.filename.replace("/", "\\")))
        # Print dirt image names for debugging
    print(f"Dirt Image Names: {dirt_image_names}",flush=True)  
    for image in uploaded_images:
        print(f"Time taken for image {image['file_path']}: {image['elapsed_time']} seconds",flush=True)

    # Pass the list of uploaded images and dirt image names to the template
    return render_template('index2.html', uploaded_images=uploaded_images, dirt_image_names=dirt_image_names)
# Ignore requests for favicon.ico
@app.route('/favicon.ico')
def favicon():
    return '', 204

# ... (existing code)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

