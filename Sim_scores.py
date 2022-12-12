from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from ast import literal_eval
from fast_text import Beds, Cupboards,Dressing,Lamps

data=[Beds, Cupboards,Dressing,Lamps]
b_embeds = list(Beds["embeds"])
c_embeds = list(Cupboards["embeds"])
d_embeds = list(Dressing["embeds"])
l_embeds = list(Lamps["embeds"])

b_embeds_feat = list(Beds["feature_embeds"])
c_embeds_feat = list(Cupboards["feature_embeds"])
d_embeds_feat = list(Dressing["feature_embeds"])
l_embeds_feat = list(Lamps["feature_embeds"])
dic={
    "Beds" : 0,
    "Cupboards":1,
    "Dressing":2,
    "Lamps":3
}
dic1={
    0:"Beds",
    1:"Cupboards",
    2:"Dressing",
    3:"Lamps"
}
embeds=[b_embeds,
c_embeds ,
d_embeds,
l_embeds ]
embeds_feat=[b_embeds_feat,
c_embeds_feat ,
d_embeds_feat,
l_embeds_feat ]
#similarity scores
def top10_score(item,item2,tag,Top_n,id):
    compares= list(x for x in range(4) if x != dic[tag])
    simi=[]
    simi2=[]
    for x in compares:
        simi.append( cosine_similarity(item.reshape(1,-1),embeds[x]))
        simi2.append( cosine_similarity(item2.reshape(1,-1),embeds_feat[x]))
    cummulative=[]
    for x,y in zip(simi,simi2):
        lis=[]
        for i,j in zip(x,y):
            lis.append(0.25*i+0.75*j)
        cummulative.append(lis)

    emb3Analyze = pd.DataFrame()
    emb3Analyze[tag] =[id for x in range(Top_n)]
    for x in range(3):
        emb=[]
        for index in cummulative[x][0].argsort()[-Top_n:][::-1]:
            emb.append(data[compares[x]][["Index","Description"]].iloc[[index]].values[0])
        emb3Analyze[dic1[compares[x]]] = emb


    return emb3Analyze  

#print(b_embeds_feat)
print(top10_score(b_embeds[38],b_embeds_feat[38],"Beds",10,"B39"))