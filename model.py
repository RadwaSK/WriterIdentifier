from sklearn.neighbors import KNeighborsClassifier 
from PIL import Image
from sklearn import svm
import numpy as np


def model(X, Y, X_test, model_type='knn'):
	if model_type == 'knn':
		knn = KNeighborsClassifier(n_neighbors = 3)
		knn.fit(X, Y)
		pred = knn.predict(X_test)

	elif model_type == 'svm':
		clf = svm.SVC()
		clf.fit(X, Y)
		pred = np.array(clf.predict(X_test))

	ones = len(pred[pred == '1'])
	twos = len(pred[pred == '2'])
	threes = len(pred[pred == '3'])
	
	prediction = np.argmax([ones, twos, threes]) + 1
	return prediction
