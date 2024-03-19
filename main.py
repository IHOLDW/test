import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import load_img, img_to_array
model = keras.models.load_model(r"model\retrainDVG.h5")

def getPrediction(filename):
    img_path = 'static/images/'+filename
    img = load_img(img_path,target_size = (256,256))
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x,axis=0) 
    p = model.predict(x)

    if p > 0.6:
        return "DOG"
    else:
        return "CAT"