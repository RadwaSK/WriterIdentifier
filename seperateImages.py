import os
import shutil

images = os.listdir('dataset')

metadata_file = open('forms.txt', 'r')
sentences = metadata_file.readlines()

metadata_info = []
for sentence in sentences:
	if sentence[0] == '#':
		continue
	sentence_list = sentence.split(' ')
	metadata_info.append([sentence_list[0], sentence_list[1]])

metadata_info =  sorted(metadata_info, key=lambda x: x[0])

for i in range(len(images)):
	image_name = images[i]
	author = metadata_info[i][1]
	if not os.path.exists('data/01/' + author):
		os.makedirs('data/01/' + author)
	shutil.copy2('data/01/' + image_name, 'data/01/' + author)

