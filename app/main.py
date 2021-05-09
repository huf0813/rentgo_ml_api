import uuid
import os
import shutil
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import cv2

app = FastAPI()
model_cnn = tf.keras.models.load_model('final_furniture_detector.h5')
label = ['table', 'chair']


def custom_response(status, message, data):
    return {'status': status, 'message': message, 'data': data}


@app.post("/predict")
def upload_skin_image(furniture_image: UploadFile = File(...)):
    generate_uuid = uuid.uuid4().hex
    file_name_uuid = '{}_{}'.format(generate_uuid, furniture_image.filename)
    with open('media/{}'.format(file_name_uuid), 'wb') as image:
        shutil.copyfileobj(furniture_image.file, image)

    img = cv2.imread('media/{}'.format(file_name_uuid))
    img = cv2.resize(img, (150, 150))
    x = np.expand_dims(img, axis=0)
    classes = model_cnn.predict(x, batch_size=10)

    os.remove('media/{}'.format(file_name_uuid))

    if classes == 0:
        return custom_response(True, furniture_image.filename, 'kursi')
    else:
        return custom_response(True, furniture_image.filename, 'meja')
