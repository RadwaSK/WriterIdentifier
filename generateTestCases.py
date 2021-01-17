import os
import shutil

authors = os.listdir('all')

# remove authors with images less than 3
for a in authors:
	if len(os.listdir('all/' + a)) < 3:
		shutil.rmtree('all/' + a)

# I made sure that remaining authors are more 300

# generate 100 test cases
test_cases_num = 100
pred = 1

correct_file = open('correct.txt', 'w')

for t in range(test_cases_num):
	ai1 = t * 3
	ai2 = ai1 + 1
	ai3 = ai2 + 1

	if t < 10:
		t_name = 'data/0' + t
	else:
		t_name = 'data/' + t

	a1 = authors[ai1]
	a2 = authors[ai2]
	a3 = authors[ai3]

	images1 = os.listdir('all/' + a1)
	images2 = os.listdir('all/' + a2)
	images3 = os.listdir('all/' + a3)

	shutil.copyfile('all/' + a1 + '/' + images1[0], t_name + '/1/' + images[0])
	shutil.copyfile('all/' + a1 + '/' + images1[1], t_name + '/1/' + images[1])

	shutil.copyfile('all/' + a2 + '/' + images2[0], t_name + '/2/' + images2[0])
	shutil.copyfile('all/' + a2 + '/' + images2[1], t_name + '/2/' + images2[1])

	shutil.copyfile('all/' + a3 + '/' + images3[0], t_name + '/3/' + images3[0])
	shutil.copyfile('all/' + a3 + '/' + images3[1], t_name + '/3/' + images3[1])

	if pred == 1:
		shutil.copyfile('all/' + a1 + '/' + images1[2], t_name + '/' + images1[2])
		os.rename(t_name + '/' + images1[2], t_name + '/test.png')
	elif pred == 2:
		shutil.copyfile('all/' + a2 + '/' + images2[2], t_name + '/' + images2[2])
		os.rename(t_name + '/' + images2[2], t_name + '/test.png')
	elif pred == 3:
		shutil.copyfile('all/' + a3 + '/' + images3[2], t_name + '/' + images3[2])
		os.rename(t_name + '/' + images3[2], t_name + '/test.png')

	correct_file.write(str(pred))
	correct_file.write('\n')

	pred += 1
	if pred == 4:
		pred = 1
