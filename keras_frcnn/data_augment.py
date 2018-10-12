import cv2
import numpy as np
import copy


def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	rgb_img = cv2.imread(img_data_aug['filepath'])

	therm_img = cv2.imread(img_data_aug['filepath'].replace('visible','lwir'))

	if augment:
		rows, cols = rgb_img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			rgb_img = cv2.flip(rgb_img, 1)
			therm_img = cv2.flip(therm_img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			rgb_img = cv2.flip(rgb_img, 0)
			therm_img = cv2.flip(therm_img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				rgb_img = np.transpose(rgb_img, (1,0,2))
				rgb_img = cv2.flip(rgb_img, 0)
				therm_img = np.transpose(therm_img, (1,0,2))
				therm_img = cv2.flip(therm_img, 0)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2
				elif angle == 0:
					pass

	img_data_aug['width'] = rgb_img.shape[1]
	img_data_aug['height'] = rgb_img.shape[0]
	return img_data_aug, rgb_img, therm_img
