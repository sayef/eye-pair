import numpy as np
import cv2
import glob
import argparse
import os
import re
import logging
import tqdm
from PIL import Image
from skimage.transform import SimilarityTransform
# global variable for mtcnn detector
mtcnn_detector = None

# global variables for cascade detector
face_cascade = None
eye_pair_cascade = None

def align(img, landmarks, image_size=(112, 112)):
	'''
	Takes oirignial image and detected landmarks as numpy format
	:param img: original image
	:param landmarks: detected landmarks, [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]
	:param image_size: output image_size
	:return warped_landmarks:
	:return warped_image:
	'''
	assert isinstance(img, np.ndarray)
	assert isinstance(landmarks, np.ndarray)
	assert landmarks.shape == (5, 2)

	M = None
	src = np.array([
		[30.2946, 51.6963],
		[65.5318, 51.5014],
		[48.0252, 71.7366],
		[33.5493, 92.3655],
		[62.7299, 92.2041]], dtype=np.float32)
	if image_size[1] == 112:
		src[:, 0] += 8.0
	dst = landmarks.astype(np.float32)
	tform = SimilarityTransform()
	tform.estimate(dst, src)
	M = tform.params[0:2, :]
	warped_image = cv2.warpAffine(img, M, image_size, borderValue=0.0)
	warped_landmarks = cv2.perspectiveTransform(landmarks[np.newaxis, ...], tform.params[0:3, :])[0]
	return warped_landmarks, warped_image

def get_eye_pair_mtcnn(image):
	
	bboxes_list, landmarks_list = mtcnn_detector.detect_faces(image)

	if not len(bboxes_list): return (False, False)
	
	# take only first occurance
	# and change landmarks array orientation 
	# from [x1, x2, x3, x4, x5, y1, y2, y3, y4, y5]
	# to [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]

	bboxes, landmarks  = bboxes_list[0], np.dstack([landmarks_list[0][:5], landmarks_list[0][5:]])

	# convert PIL `image` to numpy array and reshape `landmarks` to (5, 2)
	landmarks, aligned = align(np.asarray(image), landmarks.reshape((5, 2)))

	# flatten `landmarks` and convert back `aligned` to PIL image
	landmarks, aligned = landmarks.flatten(), Image.fromarray(aligned)

	width, height = aligned.size
	left_eye = (landmarks[0], landmarks[1])
	right_eye = (landmarks[2], landmarks[3])
	
	y_start = min(left_eye[1], right_eye[1])
	y_end = max(left_eye[1], right_eye[1])

	x_start = min(left_eye[0], right_eye[0])
	x_end = max(left_eye[0], right_eye[0])

	eye_pair = aligned.crop((x_start-width*0.20, y_start-height*0.15, x_end+width*0.20, y_end+height*0.10))
	
	return (True, eye_pair)
	
def get_eye_pair_opencv(image):
	
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for x, y, w, h in faces: # face points
		roi_gray = gray[y:y + h, x:x + w]
		roi_color = image[y:y + h, x:x + w]
		eye_pairs = eye_pair_cascade.detectMultiScale(roi_gray)

		for (ex, ey, ew, eh) in eye_pairs: # eye_pair points
			roi_eyes = roi_color[ex:min(roi_color.shape[0], ex+ew*2), max(0, ey-eh//2):min(roi_color.shape[1], ey+eh)]
			return (True, roi_eyes)
	
	return (False, False)

def save_eye_pair(input_path, output_path, method='mtcnn'):
	if method == 'mtcnn':
		image = Image.open(input_path)
		found, pair = get_eye_pair_mtcnn(image)
		if found:
			os.makedirs(os.path.split(output_path)[0], exist_ok = True)
			pair.save(output_path)
	else:
		image = cv2.imread(input_path)
		found, pair = get_eye_pair_opencv(image)
		if found:
			os.makedirs(os.path.split(output_path)[0], exist_ok = True)
			cv2.imwrite(output_path, pair)

def main(args, exts=('.jpg','.jpeg','.png')):
	if args.method == 'mtcnn':
		global mtcnn_detector
		from mtcnn import MTCNN
		mtcnn_detector = MTCNN(args.device)
	else:
		global face_cascade, eye_pair_cascade
		face_cascade = cv2.CascadeClassifier('./haar-cascade/haarcascade_frontalface_default.xml') # face 
		eye_pair_cascade = cv2.CascadeClassifier('./haar-cascade/haarcascades_haarcascade_mcs_eyepair_big.xml') #eye_pair

	files = list(filter(lambda x: x.lower().endswith(exts), glob.glob(args.input_dir + "/**/*", recursive=True)))

	for input_path in tqdm.tqdm(files):
		save_eye_pair(input_path, os.path.join(args.output_dir, re.sub(r'^{}/'.format(args.input_dir), '', input_path)), method=args.method)

# python crop_eye_pair.py --input_dir images/ --output_dir eye_pair_images/ --method mtcnn --device cuda
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='crop eye pair from face image')
	parser.add_argument('--input_dir', default='', type=str, help='')
	parser.add_argument('--output_dir', default='', type=str, help='')
	parser.add_argument('--method', default='mtcnn', type=str, help='mtcnn or cascade')
	parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
	args = parser.parse_args()

	main(args)