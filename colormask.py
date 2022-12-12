import cv2 
import numpy as np
from matplotlib import pyplot as plt

import imutils

def segment(path):
    img = cv2.imread(path)
    img2=img.copy()
    edges = cv2.Canny(img.copy(),100,200)
    image = edges
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray,135,135, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE   , cv2.RETR_TREE)
    # draw all contours
    n=len(contours)-1
    contours=sorted(contours,key=cv2.contourArea,reverse=False)[:n]
    for c in contours:
        hull=cv2.convexHull(c)
        cv2.drawContours(image,[hull],0,(0,255,0),2)
        g2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        r, t2 = cv2.threshold(g2, 140,180, cv2.THRESH_BINARY)

        masked = cv2.bitwise_and(img2, img2, mask = t2)
    return masked
