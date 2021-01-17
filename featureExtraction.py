import os
import cv2 as cv
import numpy as np
from PIL import Image

def sort_contours(cnts, method="top-to-bottom"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True	
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to bottom
	boundingBoxes = [cv.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


def extract_contours(img):
	img_inv = cv.bitwise_not(img)
	contours, hierarchy = cv.findContours(img_inv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	rects = [cv.boundingRect(c) for c in contours]
	new_conts = []
	for i, r in enumerate(rects):
		x, y, w, h = r
		if w < 60 and h < 35:
			continue
		new_conts.append(contours[i])
	return new_conts


def get_masks_edge():
    mask_list = []
    end_point_list = [(0,3),(1,3),(2,3),(3,3),(3,2),(3,1),(3,0),(3,2),(3,1),(3,0),(2,0),(1,0)]
    center         = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,3),(0,3),(0,3),(0,3),(0,3)]
    for i in range(0,len(end_point_list)):
        window = np.zeros((4,4))
        image = cv.line(window, center[i], end_point_list[i], (255,255,255), 1)
        mask_list.append(window)
    return mask_list


def edge_direction_feature(img):
    # Convert the img to grayscale 
    gray = img
    # Apply edge detection method on the image 
    edges = cv.Canny(gray,50,150,apertureSize = 3) 
    _,thershold = cv.threshold(edges,127,255,cv.THRESH_BINARY)
    im = Image.fromarray(thershold)

    masks = get_masks_edge()
    feature_vector = np.zeros(12)
    thershold = thershold.astype(np.uint8)
    for i  in range(0,12):
        mask = masks[i].astype(np.uint8)
        res = cv.matchTemplate(thershold,mask,cv.TM_CCOEFF_NORMED)
        thr = 0.8
        loc = np.where( res >= thr)
        feature_vector[i] = len(list(zip(*loc[::-1])))
        for pt in zip(*loc[::-1]):
            cv.rectangle(img, pt, (pt[0] + 4, pt[1] + 4), (0,0,255), 2)
            
    return feature_vector


def get_masks_hinge():
    mask_list = []
    end_point_list = [(3,6),(4,6),(5,6),(6,6),(6,5),(6,4),(6,3),(6,2),(6,1),(6,0),(5,0),(4,0),(3,0),(2,0),(1,0),(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(1,6),(2,6)]
    for i in range(0,12):
        for j in range(0,24):
            if(i==0 and j==23):continue
            if(i != j and i-1 != j and  i+1 != j):
                window = np.zeros((7,7))
                image = cv.line(window, (3,3), end_point_list[i], (255,255,255), 1)
                image = cv.line(window, (3,3), end_point_list[j], (255,255,255), 1)
                mask_list.append(window)
    return mask_list


def edge_hinge_feature(img):
    # Convert the img to grayscale 
    gray = img
  
    # Apply edge detection method on the image 
    edges = cv.Canny(gray,50,150,apertureSize = 3) 
    _,thershold = cv.threshold(edges,127,255,cv.THRESH_BINARY)
    
    im = Image.fromarray(thershold)

    masks = get_masks_hinge()
    feature_vector = np.zeros(252)
    thershold = thershold.astype(np.uint8)
    for i  in range(0,252):
        mask = masks[i].astype(np.uint8)
        res = cv.matchTemplate(thershold,mask,cv.TM_CCOEFF_NORMED)
        thr = 0.8
        loc = np.where( res >= thr)
        feature_vector[i] = len(list(zip(*loc[::-1])))
        for pt in zip(*loc[::-1]):
            cv.rectangle(img, pt, (pt[0] + 4, pt[1] + 4), (0,0,255), 2)
            
    return feature_vector


def remove_spaces(img):
	_, img_binary = cv.threshold(img, 180, 255, 0)

	kernel = np.array([[1, 1, 1, 1, 1],[1, 1, 1, 1, 1],[1, 1, 1, 1, 1]])
	img_eroded = cv.erode(img_binary, kernel, iterations=3)
	# first get contours
	contours = extract_contours(img_eroded)
	# sort from left to right
	contours, rects = sort_contours(contours, 'left-to-right')
	new = np.ones((img.shape[0],img.shape[1]*2))*255
	prevx=0
	prevy=0
	for i in range(len(rects)):
	    r = rects[i]
	    x, y, w, h = r
	    new[prevy:prevy+h,prevx:prevx+w]= img[y:y+h,x:x+w]
	    prevx=prevx+w
	    prevy=0
	return new


def LBP(img):
	xi = np.zeros(256, dtype='float64')
	gray = remove_spaces(img)    
	lbp = np.zeros(gray.shape)
	coords = np.column_stack(np.where(gray < 100))
	step = 3
	for i,j in coords:
		if(i < gray.shape[0]-step and j < gray.shape[1]-step):
			neigh = [(i,j+step),(i,j-step),(i+step,j),(i-step,j),(i+step,j+step),(i+step,j-step),(i-step,j+step),(i-step,j-step)]
			test_list = [int(gray[i][j] <= gray[x[0]][x[1]]) for x in neigh]
			lbp[i][j] = int("".join(str(x) for x in test_list), 2)
			xi[int(lbp[i][j])] += 1
	return xi


def extract_features(img, features_type):
	if features_type == 'white':
		xi = np.zeros(4, dtype='float64')

		if img is None:
			return xi

		height, width = img.shape
		
		_, img_binary = cv.threshold(img, 180, 255, 0)
		
		# extract feature 1: white spacing
		#first dialate by width
		kernel = np.array([[1, 1, 1, 1, 1],
							[1, 1, 1, 1, 1],
							[1, 1, 1, 1, 1]])
		img_eroded = cv.erode(img_binary, kernel, iterations=3)
		# first get contours
		contours = extract_contours(img_eroded)
		# sort from left to right
		contours, rects = sort_contours(contours, 'left-to-right')
		
		differences_list = []
		last_end = 0
		for i in range(len(rects)):
			r = rects[i]
			x, y, w, h = r
			if i != 0:
				diff = x - last_end
				if diff < 20:
					last_end = x + w
					continue
				else:
					differences_list.append(diff)
			last_end = x + w

		if len(differences_list) > 0:
			xi[0] = feautre1 = np.median(differences_list)
		else:
			xi[0] = 0


		# extract feature 2, 3 and 4 (ratio of ranges between baselines)
		count = []
		for i in range(img_binary.shape[0]):
		    count.append(0)
		    for j in range(img_binary.shape[1]):
		        if img_binary[i][j] == 0:
		            count[i] = count[i]+1
		
		mini = 0
		maxi = 0
		avg = sum(count) / len(count)
		upper =0
		lower =0

		for i in range(len(count)):
		    if count[i] > 0 and mini == 0:
		        mini = i
		    if count[i] > avg:
		        upper = i
		        break

		i = img_binary.shape[0] - 1        
		while i > 0:
		    if count[i] > 0 and maxi == 0:
		        maxi = i
		    if count[i] > avg:
		        lower = i
		        break
		    i = i - 1
		
		epsilon = 0.000001
		f1 = abs(mini - upper) + epsilon
		f2 = abs(upper - lower) + epsilon
		f3 = abs(lower - maxi) + epsilon

		xi[1] = feature2 = f1 / f2
		xi[2] = feature3 = f1 / f3
		xi[3] = feature4 = f2 / f3	

	elif features_type == 'edge':
		xi = np.zeros(12)
		if img is None:
			return xi
		xi = edge_direction_feature(img)

	elif features_type == 'hinge':
		xi = np.zeros(252)
		if img is None:
			return xi
		xi = edge_hinge_feature(img)

	elif features_type == 'lbp':
		xi = np.zeros(256)
		if img is None:
			return xi
		xi = LBP(img)

	return xi
	

def features(authors_images, test_images, authorsIDs, features_type='edge'):
	X = []
	Y = []
	X_test = []

	for i, a in enumerate(authors_images):
		auth_ID = authorsIDs[i]		
		for img in a:
			X.append(extract_features(img, features_type)) # array of size 4
			Y.append(auth_ID)
	
	if len(test_images) == 0:
		if features_type == 'white':
			X_test = np.zeros(4)
		elif features_type == 'edge':
			X_test = np.zeros(12)
		elif features_type == 'hinge':
			X_test = np.zeros(252)
		elif features_type == 'lbp':
			X_test = np.zeros(256)

	for img in test_images:
		X_test.append(extract_features(img, features_type))	

	return X, Y, X_test

