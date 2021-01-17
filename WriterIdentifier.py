from preprocessing import *
from featureExtraction import *
from model import *
import time


def readImages(test_case_path):
	authors = os.listdir(test_case_path)
	sorted(authors)

	authors_forms_images = []
	IDs = []

	for a in authors:
		if a.find('.') != -1: # test.png
			test_img_path = test_case_path + '/' + a
			test_form_image = cv.imread(test_img_path, cv.IMREAD_GRAYSCALE)
			break

		IDs.append(a)
		a = test_case_path + '/' + a
		forms = os.listdir(a)

		imgs = []

		for fi, f in enumerate(forms):
			# break them into lines, save them into lines_path
			full_path = a + '/' + f
			img = cv.imread(full_path, cv.IMREAD_GRAYSCALE)
			imgs.append(img)
			
		
		authors_forms_images.append(imgs)

	return np.array(authors_forms_images), np.array(test_form_image), IDs


if __name__ == '__main__':
	test_cases = os.listdir('data')
	sorted(test_cases)

	time_file = open('time.txt', 'w')
	results_file = open('results.txt', 'w')

	for t in test_cases:
		test_case_path = 'data/' + t
		
		authors_forms_images, test_form_image, IDs = readImages(test_case_path)
		
		start = time.time()
		
		authors_lines_images, test_lines_image = preprocess(authors_forms_images, test_form_image)
		X, Y, X_test = features(authors_lines_images, test_lines_image, IDs, features_type='edge')
		prediction = model(X, Y, X_test, model_type='knn')

		end = time.time()

		duration = end - start

		time_file.write(str(duration) + '\n')
		results_file.write(str(prediction) + '\n')

	time_file.close()
	results_file.close()
