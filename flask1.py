import os
from PIL import Image
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.applications.vgg16 import decode_predictions
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.backend import get_session

    
    

from flask import Flask, render_template, request
app = Flask(__name__)  
@app.route('/', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f1 = request.files['file']
        f1.save("qry1.png")
        file_name='qry1.png'
        X_i = os.path.join(file_name)
        print(X_i)
        result=predict(file_name)
        return render_template("result.html",var=result,img_path=X_i) 
    
    return render_template("index.html") 

def predict(data_file):
    model = tf.keras.models.load_model('FlowerClassification.model')
    IMG_SHAPE = (224, 224, 1)
    #X_i = image.open(os.path.join(data_file))
    #img1 = cv2.imread(X_i,0)/255
    #img1 = cv2.resize(img1, (200,200))
    #img1=img1.reshape(IMG_SHAPE)
    img=tf.keras.utils.load_img(data_file,target_size=(224,224))
    img = tf.keras.utils.img_to_array(img)/255
    img=np.array([img])
    img.shape
    result = model.predict(img)
    a=np.argmax(result)
    if a == 0:
        ans = 'covid'
    elif a == 1:
        ans = 'normal'
    else:
        ans = 'pneumonia'
    return(ans)
    
if __name__ == '__main__':
    app.run(debug=False)
    