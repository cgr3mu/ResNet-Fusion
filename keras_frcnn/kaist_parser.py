import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
def get_data(input_path,mode):
	all_imgs = []

	classes_count = {}

	class_mapping = {}

	visualize = False

	#data_paths = input_path #[os.path.join(input_path,s) for s in ['VOC2007', 'VOC2012']]
	print('Parsing annotation files')
	
	train_imgs_path = os.path.join(input_path,'Train.txt')
	test_imgs_path = os.path.join(input_path,'Test.txt')
	
	#for data_path in data_paths:

		#imgsets_path_trainval = os.path.join(data_path,'train.txt')
		#imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','test.txt')
		#annot_path = os.path.join(data_path, 'Annotations')
		#imgs_path = os.path.join(data_path, 'JPEGImages')
	
	if mode == "train":
		data_path = train_imgs_path
	if mode == "test":
		data_path = test_imgs_path
		
	trainval_files = []
	test_files = []		
	idx = 0
	with open(data_path, 'r') as ft:
		for line in ft:
			line  = line.strip("\n")
			
			annot = line.replace("images","annotations-xml")
			annot = annot.replace(".jpg",".xml")
			annot = annot.replace("/visible","/")
		
			idx += 1

			et = ET.parse(annot)
			element = et.getroot()

			element_objs = element.findall('object')
			element_filename = element.find('filename').text
			element_width = int(element.find('size').find('width').text)
			element_height = int(element.find('size').find('height').text)

			if len(element_objs) > 0:
				annotation_data = {'filepath': line, 'width': element_width,
								   'height': element_height, 'bboxes': []}

				'''if element_filename in trainval_files:
					annotation_data['imageset'] = 'trainval'
				elif element_filename in test_files:
					annotation_data['imageset'] = 'test'
				else:
					annotation_data['imageset'] = 'trainval'''

				for element_obj in element_objs:
					#class_name = element_obj.find('name').text
					class_name = "person"
					if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1
					class_mapping[class_name] = 0
					'''if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1

					if class_name not in class_mapping:
						class_mapping[class_name] = len(class_mapping)'''

					obj_bbox = element_obj.find('bndbox')
					x1 = int(round(float(obj_bbox.find('xmin').text)))
					y1 = int(round(float(obj_bbox.find('ymin').text)))
					x2 = int(round(float(obj_bbox.find('xmax').text)))
					y2 = int(round(float(obj_bbox.find('ymax').text)))
					difficulty = int(element_obj.find('difficult').text) == 1
					#print([x1,y1,x2,y2])
					annotation_data['bboxes'].append({'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
				all_imgs.append(annotation_data)
			else:
				continue

			if visualize:
				img = cv2.imread(annotation_data['filepath'])
				for bbox in annotation_data['bboxes']:
					cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
								  'x2'], bbox['y2']), (0, 0, 255))
				cv2.imshow('img', img)
				cv2.waitKey(0)

	return all_imgs, classes_count, class_mapping
	
'''if __name__ == "__main__":
	input_path = "/home/kishan/Documents/Knowledge_distillation_ped_detection/github_clone/rgbt-ped-detection-master/data/kaist-rgbt/"
	imgs_dicts = get_data(input_path)
	print imgs_dicts[59]
	print len(imgs_dicts)'''
	
	
