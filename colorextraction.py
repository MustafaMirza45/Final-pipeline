from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2
from colormask import segment
import webcolors

from sklearn.metrics import mean_squared_error

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        hex_color += ("{:02x}".format(int(i)))
    return hex_color
def preprocess(raw):
    image = cv2.resize(raw, (299, 299), interpolation = cv2.INTER_AREA)
    image = image.reshape(image.shape[0]*image.shape[1], 3)
    return image
def analyze(img,a):
    clf = KMeans(n_clusters = 5)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    plt.figure(figsize = (12, 8))
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    #print("Found the following colors:\n")
    plt.savefig(a+".png")
    lis={}
    for color,y in zip(hex_colors,counts.values()):
     # print(color,"  ",y)
      lis[y]=color
    
    return lis
def hex2name(c):
    h_color = c
    try:
        nm = webcolors.hex_to_name(h_color, spec='css3')
    except ValueError as v_error:
        print("{}".format(v_error))
        rms_lst = []
        for img_clr, img_hex in webcolors.CSS3_NAMES_TO_HEX.items():
            cur_clr = webcolors.hex_to_rgb(img_hex)
            rmse = np.sqrt(mean_squared_error(webcolors.hex_to_rgb(c), cur_clr))
            rms_lst.append(rmse)

        closest_color = rms_lst.index(min(rms_lst))

        nm = list(webcolors.CSS3_NAMES_TO_HEX.items())[closest_color][0]
    return nm
def color(path):
    masked=segment(path)
    masked1 = cv2.cvtColor(masked.copy(), cv2.COLOR_BGR2RGB)
    modified_masked = preprocess(masked1)
    lis1 = analyze(modified_masked,"a")
    image = cv2.imread(path)
    image2 = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    modified_image = preprocess(image2)
    lis = analyze(modified_image,"b")
    top=sorted(lis1.keys(),reverse=True)
    top2=sorted(lis.keys(),reverse=True)
    
    background=hex2name(lis[top2[0]])
    object=hex2name(lis1[top[1]])
    #print("background = ",background)
    #print("object = ",object)
    return object,background

#color("E:\FYP\Final pipeline\Beds\B2.jpg")