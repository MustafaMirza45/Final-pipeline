from asyncore import loop
from pyexpat import features
from typing import List
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
from Sim_scores import top10_score
from fast_text import fast
from fast_text import Beds, Cupboards,Dressing,Lamps
import os
import pickle
haralick= pickle.load(open("Haralicks.pkl", "rb"))

b_index = list(Beds["Index"])
c_index = list(Cupboards["Index"])
d_index = list(Dressing["Index"])
l_index = list(Lamps["Index"])

b_embeds = list(Beds["embeds"])
c_embeds = list(Cupboards["embeds"])
d_embeds = list(Dressing["embeds"])
l_embeds = list(Lamps["embeds"])

b_embeds_feat = list(Beds["feature_embeds"])
c_embeds_feat = list(Cupboards["feature_embeds"])
d_embeds_feat = list(Dressing["feature_embeds"])
l_embeds_feat = list(Lamps["feature_embeds"])

data=["Beds", "Cupboards","Dressing","Lamps"]
dic={
    "Beds" : 0,
    "Cupboards":1,
    "Dressing":2,
    "Lamps":3
}
embeds=[b_embeds,
c_embeds ,
d_embeds,
l_embeds ]
embeds_feat=[b_embeds_feat,
c_embeds_feat ,
d_embeds_feat,
l_embeds_feat ]
index=[b_index,
c_index ,
d_index,
l_index ]
df=pd.DataFrame(columns=["Beds","Cupboards","Dressing","Lamps"])

for x in data:
    for y in range(10):
        df=df.append(top10_score(embeds[dic[x]][y],embeds_feat[dic[x]][y],x,10,index[dic[x]][y]),ignore_index=True)

for index,rows in df.iterrows():
    #print(type(rows["Beds"]))
    if type(rows["Beds"]) == type(np.ndarray(1)):
        df.loc[index,"Beds"]=rows["Beds"][0]
    if type(rows["Cupboards"]) == type(np.ndarray(1)):
        df.loc[index,"Cupboards"]=rows["Cupboards"][0]
    if type(rows["Dressing"]) == type(np.ndarray(1)):
        df.loc[index,"Dressing"]=rows["Dressing"][0]
    if type(rows["Lamps"]) == type(np.ndarray(1)):
        df.loc[index,"Lamps"]=rows["Lamps"][0]
df["rating"]=[1 for x in range(400)]
df["review"]=[0 for x in range(400)]
df["odd"]=["" for x in range(400)]
df.to_csv("pipeline_pairings.csv")
a="y"
while a.lower()=="y": 
    if not os.path.exists('input'):
        os.mkdir('input')
    print("Do you want to input another picture? y/n")
    a=input()
    if a.lower() != "y":
        break
    print("copy a single picture in input folder then press y")
    input()
    img_path=os.listdir("E:\FYP\Final pipeline\Input")
    name= img_path[0].split("\\")[-1]
    path=img_path[0]
    mat=material("E:\FYP\Final pipeline\Input\\"+path,haralick)
    obj,background=color("E:\FYP\Final pipeline\Input\\"+path)
    features = mat +" "+obj+" "+background
    print("your image is : \n1.Bed \n2.Cupboard\n3.Dressing\n4.Lamp")
    tag=int(input())-1
    feat="y"
    features2=""
    while feat.lower() == "y":
        print("input additional tags: ")
        features2= features2+" "+ input()
        print("do you want to specify more tags? y/n")
        feat=input()
    embed=fast(features)
    embed2=fast(features2)
    guessdf=top10_score(embed,embed2,data[tag],10,name)
    
    for index,rows in guessdf.iterrows():
        #print(type(rows["Beds"]))
        if type(rows["Beds"]) == type(np.ndarray(1)):
            guessdf.loc[index,"Beds"]=rows["Beds"][0]+".jpg"
        if type(rows["Cupboards"]) == type(np.ndarray(1)):
            guessdf.loc[index,"Cupboards"]=rows["Cupboards"][0]+".jpg"
        if type(rows["Dressing"]) == type(np.ndarray(1)):
            guessdf.loc[index,"Dressing"]=rows["Dressing"][0]+".jpg"
        if type(rows["Lamps"]) == type(np.ndarray(1)):
            guessdf.loc[index,"Lamps"]=rows["Lamps"][0]+".jpg"
    for index,row in guessdf.iterrows():
        fig = plt.figure(figsize=(10, 7))

        rows = 2
        columns = 2
        Image1 = cv2.imread("Beds\\"+str(row["Beds"]))
        Image1 = cv2.cvtColor(Image1, cv2.COLOR_BGR2RGB)
        Image2 = cv2.imread("Cupboards\\"+str(row["Cupboards"]))
        Image2 = cv2.cvtColor(Image2, cv2.COLOR_BGR2RGB)
        Image3 = cv2.imread("Dressing\\"+str(row["Dressing"]))
        Image3 = cv2.cvtColor(Image3, cv2.COLOR_BGR2RGB)
        Image4 = cv2.imread("Lamps\\"+str(row["Lamps"]))
        Image4 = cv2.cvtColor(Image4, cv2.COLOR_BGR2RGB)
        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)
        
        # showing image
        plt.imshow(Image1)
        plt.axis('off')
        plt.title("First")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)
        
        # showing image
        plt.imshow(Image2)
        plt.axis('off')
        plt.title("Second")
        
        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 3)
        
        # showing image
        plt.imshow(Image3)
        plt.axis('off')
        plt.title("Third")
        
        # Adds a subplot at the 4th position
        fig.add_subplot(rows, columns, 4)
        
        # showing image
        plt.imshow(Image4)
        plt.axis('off')
        plt.title("Fourth")
        plt.savefig('pairings '+str(index)+'.png')
        input("press enter for next")
        