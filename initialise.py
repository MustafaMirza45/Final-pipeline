import random
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Function to Extract features from the images

def image_feature(direc,n):
    model = tf.keras.models.load_model('tfresnet50.model')
    features = [];
    img_name = [];
    for i in tqdm(direc):
        try:
            fname=n +'/'+i
            img=image.load_img(fname,target_size=(299,299))
            x = img_to_array(img)
            x=np.expand_dims(x,axis=0)
            x=preprocess_input(x)
            feat=model.predict(x)
            feat=feat.flatten()
            features.append(feat.tolist())
            img_name.append(i)
            #shutil.move(os.path.join('cluster', i), 'cluster2')
        except:
            print("bad image")
    return features,img_name
#img_path=os.listdir("E:\FYP\module 1\Train Data\cluster")
#print(img_path)
#random.shuffle(img_path)
#img_features,img_name=image_feature(img_path,'Train Data\cluster')


#print(type(img_features))
#print(type(img_features[0]))