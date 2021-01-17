import numpy as np
import cv2 as cv
import os
import shutil

def get_start_end(img):
	h, w = img.shape
	start = -1
	for i in range(w):
		for j in range(h):
			p = img[j][i]
			if start == -1 and p == 255:
				start = i
				break

		if start != -1:
			break

	end = start
	j = w - 1
	while j > start:
		i = h - 1
		while i >= 0:
			p = img[i][j]
			if p == 255:
				end = j
				break
			i -= 1
		if end != start:
			break
		j -= 1
	return start, end


def extract_lines(img):
	if img is None:
		return

	_, img_binary_orig = cv.threshold(img, 190, 255, 0)
	img_binary = cv.bitwise_not(img_binary_orig)
	length, width = img_binary.shape
	
	kernel = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
	img_dilated = cv.dilate(img_binary, kernel, iterations=10)
	
	contours, hierarchy = cv.findContours(img_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	rects = [cv.boundingRect(ctr) for ctr in contours]
	chosen_rects = []
	start = False

	rects = sorted(rects, key=lambda x: x[1])

	for r in rects:
		x, y, w, h = r
		lower_limit = length * 0.16
		upper_limit = length * 0.85
		if h < 30 and w > 350 and y > 0.1 * length:
			start = True
			continue
		if start or y > 0.25 * length:
			if y < lower_limit or y > upper_limit or h < 80 or h > 235 or w < 1000:
				continue
			chosen_rects.append(r)
		
	imgs = []

	for i, r in enumerate(chosen_rects):
		x, y, w, h = r
		new_img = img[y:y+h, x:x+w]
		start, end = get_start_end(img_binary[y:y+h, x:x+w])
		new_img = new_img[:, start:end]
		imgs.append(new_img)
		
	return imgs


def preprocess(authors_forms_images, test_form_image):
	authors_lines_images = []

	for a in authors_forms_images:
		imgs = []

		for i, img in enumerate(a):
			# break them into lines, save them into lines_path
			if i == 0:
				imgs = extract_lines(img)
			else:
				imgs = imgs + extract_lines(img)
		
		authors_lines_images.append(np.array(imgs))

	test_lines_images = extract_lines(test_form_image)

	return np.array(authors_lines_images), np.array(test_lines_images)


