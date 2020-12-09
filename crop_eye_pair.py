import numpy as np
import cv2
import glob
import argparse
import os
import re
import logging
import tqdm
from PIL import Image

# global variable for mtcnn detector
mtcnn_detector = None

# global variables for cascade detector
face_cascade = None
eye_pair_cascade = None

def get_eye_pair_mtcnn(image):
	
	bboxes, keypoints = mtcnn_detector.detect_faces(image)
	
	if not len(bboxes): return (False, False)

	left_eye = (keypoints[0][0], keypoints[0][5])
	right_eye = (keypoints[0][1], keypoints[0][6])
	
	y_start = min(left_eye[1], right_eye[1])
	y_end = max(left_eye[1], right_eye[1])

	x_start = min(left_eye[0], right_eye[0])
	x_end = max(left_eye[0], right_eye[0])

	eye_pair = image.crop((x_start-15, y_start-10, x_end+15, y_end+10))
	
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

def main(args):

	if args.device == 'cuda':
		os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	else:
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	if args.method == 'mtcnn':
		global mtcnn_detector
		from mtcnn import MTCNN
		mtcnn_detector = MTCNN(args.device)
	else:
		global face_cascade, eye_pair_cascade
		face_cascade = cv2.CascadeClassifier('./haar-cascade/haarcascade_frontalface_default.xml') # face 
		eye_pair_cascade = cv2.CascadeClassifier('./haar-cascade/haarcascades_haarcascade_mcs_eyepair_big.xml') #eye_pair

	for input_path in tqdm.tqdm(glob.glob(args.input_dir + "/**/*.jpg", recursive=True)):
		save_eye_pair(input_path, os.path.join(args.output_dir, re.sub('^' + args.input_dir, '', input_path)), method=args.method)


# python crop_eye_pair.py --input_dir images/ --output_dir eye_pair_images/ --method mtcnn --device cuda
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='crop eye pair from face image')
	parser.add_argument('--input_dir', default='', type=str, help='')
	parser.add_argument('--output_dir', default='', type=str, help='')
	parser.add_argument('--method', default='mtcnn', type=str, help='mtcnn or cascade')
	parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
	args = parser.parse_args()

	main(args)