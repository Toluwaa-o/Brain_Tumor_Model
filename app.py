import os
import cv2
import math
import numpy as np
from flask_cors import CORS
from tensorflow import keras
from skimage.transform import resize
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
model = keras.models.load_model('Brain_Tumor_Model.keras')

@app.route('/')
def test():
    return jsonify({'/classify': "Use this route to submit images, the response is the classification."})

@app.route('/classify', methods=['POST'])
def submit_and_classify():
    """
        Process an uploaded brain scan image to classify whether it contains a tumor or not.

        This function expects a POST request with a file upload containing a brain scan image.
        It saves the uploaded image to a predefined directory, preprocesses it for classification,
        and then uses a pre-trained machine learning model to predict whether the brain scan
        contains a tumor or not.

        Returns:
            - If successful:
                - A message indicating whether the brain scan contains a tumor or not.
                - HTTP status code 200 (OK).
            - If the image is not provided in the request:
                - A JSON response indicating that an image needs to be submitted.
                - HTTP status code 400 (Bad Request).
            - If there is an error during image processing:
                - A message indicating the error.
                - HTTP status code 500 (Internal Server Error).
        """

    try:
        image = request.files.get('img')
        if not image:
            return jsonify({'failed': 'Please submit an image'}), 400

        img_name = secure_filename(image.filename)
        image.save('./public/image/{}'.format(img_name))

        new_img = cv2.imread('./public/image/{}'.format(img_name))
        if new_img is None:
            return jsonify(({"failed": "Failed to read image file"})), 500

        new_img = resize(new_img, (128, 128, 3))
        new_img = np.expand_dims(new_img, axis=0)

        pred = model(new_img)
        classification = math.ceil(pred[0][0]/100)

        return 'This brain scan contains a tumor' if classification == 1 else 'This brain scan does not contain a tumor', 200
    except Exception as e:
        return "An error occurred during image processing: {}".format(e), 500

if __name__ == '__main__':
    port = os.environ.get('PORT')
    if port:
        port = int(port)
    else:
        port = 5000

    app.run(debug=True, port=port)