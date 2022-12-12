from webbrowser import BackgroundBrowser
import pandas as pd
import glob
import numpy as np
import re
from tqdm import tqdm
import nltk
from gensim.models.fasttext import FastText
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from colorextraction import color
from haralicks import material
import numpy as np
import cv2
import pickle
haralick= pickle.load(open("Haralicks.pkl", "rb"))

data=["Beds","Cupboards","Lamps","Dressing"]


for x in data:
    df=pd.read_csv(x+".csv",index_col=0)
    for index, row in df.iterrows():
        mat=material(x+"\\"+index+".jpg",haralick)
        obj,background=color(x+"\\"+index+".jpg")
        df.loc[index,"Description"]=mat+" "+obj+" "+background+" "+ df.loc[index,"Description"]
        print(mat,obj,background,index)
    df.to_csv(x+"(ext).csv")