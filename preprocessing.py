import numpy as np
import cv2 as cv
import os
import shutil

# def resize_image(image, width=None, height=None, inter=cv.INTER_AREA):
# 	dim = None
# 	(h, w) = image.shape[:2]
# 	if width is None and height is None:
# 		return image
# 	if width is None:
# 		r = height / float(h)
# 		dim = (int(w * r), height)
# 	else:
# 		r = width / float(w)
# 		dim = (width, int(h * r))
# 	resized = cv.resize(image, dim, interpolation=inter)
# 	return resized


def extract_lines(img, path, indx):
	if img is None or path is None:
		return

	_, img_binary_orig = cv.threshold(img, 180, 255, 0)
	img_binary = cv.bitwise_not(img_binary_orig)
	length, width = img_binary.shape
	
	kernel = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
						[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
	img_dilated = cv.dilate(img_binary, kernel, iterations=10)
	
	contours, hierarchy = cv.findContours(img_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	rects = [cv.boundingRect(ctr) for ctr in contours]
	chosen_rects = []
	
	for r in rects:
		x, y, w, h = r
		lower_limit = length * 0.16
		upper_limit = length * 0.85
		if y < lower_limit or y > upper_limit or h < 50 or h > 235 or w < 750:
			continue
		chosen_rects.append(r)
		
	chosen_rects = sorted(chosen_rects, key=lambda x: x[1])
	
	for i, r in enumerate(chosen_rects):
		# if i == len(chosen_rects) - 1:
		# 	continue
		x, y, w, h = r
		new_img = img[y:y+h, x:x+w]
		new_img_path = path + '/' + str(indx) + '.jpg'
		indx += 1
		cv.imwrite(new_img_path, new_img)

	return indx


test_cases = os.listdir('data')
N = len(test_cases)
i_test = 0

while (i_test != N):
	test_case = 'data/' + test_cases[i_test]
	authors_folders = np.array(os.listdir(test_case))
	
	for a in authors_folders:
		if a.find('.') != -1: # I finished folders, and now will read the test file
			test_lines_path = test_case + '/testcase_lines'

			if not os.path.exists(test_lines_path):
				os.makedirs(test_lines_path)
			
			test_img_path = test_case + '/' + a
			img = cv.imread(test_img_path, cv.IMREAD_GRAYSCALE)
			extract_lines(img, test_lines_path, 0)
			continue

		a = test_case + '/' + a
		forms = os.listdir(a)
		lines_path = a + '/lines'

		if not os.path.exists(lines_path):
			os.makedirs(lines_path)
		else:
			shutil.rmtree(lines_path)
			os.makedirs(lines_path)

		lines_count = 0

		for f in forms:
			if f == 'lines':
				continue
			# break them into lines, save them into lines_path
			full_path = a + '/' + f
			img = cv.imread(full_path, cv.IMREAD_GRAYSCALE)
			lines_count = extract_lines(img, lines_path, lines_count)
		
	i_test += 1
