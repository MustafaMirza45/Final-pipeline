import pickle
import cv2
import mahotas as mt


def extract_features(image):
    	# calculate haralick texture features for 4 types of adjacency
	textures = mt.features.haralick(image)

	# take the mean of it and return it
	ht_mean  = textures.mean(axis=0)
	return ht_mean
def material(path,haralick):
    img=cv2.imread(path)
    

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    # extract haralick texture from the image
    features = extract_features(gray)
    predict=haralick.predict(features.reshape(1, -1))

    return predict[0]
