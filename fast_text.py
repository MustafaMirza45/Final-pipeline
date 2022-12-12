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
import fasttext
import nltk
from nltk.tokenize import word_tokenize


Beds = pd.read_csv("Beds(ext).csv")
Cupboards = pd.read_csv("Cupboards(ext).csv")
Dressing = pd.read_csv("Dressing(ext).csv")
Lamps = pd.read_csv("Lamps(ext).csv")

all_titles = list(Beds["Description"])
all_titles2 = list(Cupboards["Description"])
all_titles3 = list(Dressing["Description"])
all_titles4 = list(Lamps["Description"])


np.savetxt(r'textualdata.txt', Beds['Description'].values, fmt='%s')
model_use=fasttext.train_unsupervised('textualdata.txt', dim=100)
model_use.save_model('fasttet_model_use.bin')


step3_df1 = pd.DataFrame()
step3_df2 = pd.DataFrame()
step3_df3 = pd.DataFrame()
step3_df4 = pd.DataFrame()

step3_df1["Description"] = Beds["Description"].copy()
step3_df2["Description"] = Cupboards["Description"].copy()
step3_df3["Description"] = Dressing["Description"].copy()
step3_df4["Description"] = Lamps["Description"].copy()

stopwords = nltk.corpus.stopwords.words("english")                                                                                                       
step2_embeds1 = list()
step2_embeds2 = list()
step2_embeds3 = list()
step2_embeds4 = list()
for row in step3_df1["Description"]:
  text_tokens = word_tokenize(row)
  remove_sw = [word for word in text_tokens if not word in stopwords]
  emb1 = [model_use.get_word_vector(x) for x in remove_sw]
  avg_emb1 = np.mean(emb1, axis=0)
  # listToStr = ' '.join(map(str, remove_sw))
  step2_embeds1.append(avg_emb1)
  #print(type(avg_emb1))


for row in step3_df2["Description"]:
  text_tokens = word_tokenize(row)
  remove_sw = [word for word in text_tokens if not word in stopwords]
  emb2 = [model_use.get_word_vector(x) for x in remove_sw]
  avg_emb2 = np.mean(emb2, axis=0)
  step2_embeds2.append(avg_emb2)

for row in step3_df3["Description"]:
  text_tokens = word_tokenize(row)
  remove_sw = [word for word in text_tokens if not word in stopwords]
  emb3 = [model_use.get_word_vector(x) for x in remove_sw]
  avg_emb3 = np.mean(emb3, axis=0)
  step2_embeds3.append(avg_emb3)

for row in step3_df4["Description"]:
  text_tokens = word_tokenize(row)
  remove_sw = [word for word in text_tokens if not word in stopwords]
  emb4 = [model_use.get_word_vector(x) for x in remove_sw]
  avg_emb4 = np.mean(emb4, axis=0)
  step2_embeds4.append(avg_emb4)

Beds["embeds"] = step2_embeds1
Cupboards["embeds"] = step2_embeds2
Dressing["embeds"] = step2_embeds3
Lamps["embeds"] = step2_embeds4


step3_df1 = pd.DataFrame()
step3_df2 = pd.DataFrame()
step3_df3 = pd.DataFrame()
step3_df4 = pd.DataFrame()

step3_df1["features"] = Beds["features"].copy()
step3_df2["features"] = Cupboards["features"].copy()
step3_df3["features"] = Dressing["features"].copy()
step3_df4["features"] = Lamps["features"].copy()

stopwords = nltk.corpus.stopwords.words("english")                                                                                                       
step2_embeds1 = list()
step2_embeds2 = list()
step2_embeds3 = list()
step2_embeds4 = list()
for row in step3_df1["features"]:
  text_tokens = word_tokenize(row)
  remove_sw = [word for word in text_tokens if not word in stopwords]
  emb1 = [model_use.get_word_vector(x) for x in remove_sw]
  avg_emb1 = np.mean(emb1, axis=0)
  # listToStr = ' '.join(map(str, remove_sw))
  step2_embeds1.append(avg_emb1)


for row in step3_df2["features"]:
  text_tokens = word_tokenize(row)
  remove_sw = [word for word in text_tokens if not word in stopwords]
  emb2 = [model_use.get_word_vector(x) for x in remove_sw]
  avg_emb2 = np.mean(emb2, axis=0)
  step2_embeds2.append(avg_emb2)

for row in step3_df3["features"]:
  text_tokens = word_tokenize(row)
  remove_sw = [word for word in text_tokens if not word in stopwords]
  emb3 = [model_use.get_word_vector(x) for x in remove_sw]
  avg_emb3 = np.mean(emb3, axis=0)
  step2_embeds3.append(avg_emb3)

for row in step3_df4["features"]:
  text_tokens = word_tokenize(row)
  remove_sw = [word for word in text_tokens if not word in stopwords]
  emb4 = [model_use.get_word_vector(x) for x in remove_sw]
  avg_emb4 = np.mean(emb4, axis=0)
  step2_embeds4.append(avg_emb4)

Beds["feature_embeds"] = step2_embeds1
Cupboards["feature_embeds"] = step2_embeds2
Dressing["feature_embeds"] = step2_embeds3
Lamps["feature_embeds"] = step2_embeds4
data=[Beds,Cupboards,Lamps,Dressing]
data1=["Beds(embed).csv","Cupboards(embed).csv","Lamps(embed).csv","Dressing(embed).csv"]

def fast(row):
  text_tokens = word_tokenize(row)
  remove_sw = [word for word in text_tokens if not word in stopwords]
  emb2 = [model_use.get_word_vector(x) for x in remove_sw]
  avg_emb2 = np.mean(emb2, axis=0)
  return avg_emb2