import numpy as np
import cv2
import glob
import argparse
import os
import re
import logging
import tqdm

# global variable for mtcnn detector
mtcnn_detector = None

# global variables for cascade detector
face_cascade = None
eye_pair_cascade = None

def get_eye_pair_mtcnn(image):
	
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	result = mtcnn_detector.detect_faces(image_rgb)

	if not result: return (False, False)

	bounding_box = result[0]['box']
	keypoints = result[0]['keypoints']

	left_eye = keypoints['left_eye']
	right_eye = keypoints['right_eye']

	y_start = min(left_eye[1], right_eye[1])
	y_end = max(left_eye[1], right_eye[1])

	x_start = min(left_eye[0], right_eye[0])
	x_end = max(left_eye[0], right_eye[0])

	image = image[y_start-15:y_end+15, x_start-10:x_end+10]
	
	return (True, image)
	
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
	image = cv2.imread(input_path)
	if method == 'mtcnn':
		found, pair = get_eye_pair_mtcnn(image)
	else:
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
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR
		logging.getLogger('tensorflow').setLevel(logging.ERROR)
		from mtcnn import MTCNN
		mtcnn_detector = MTCNN()
	else:
		global face_cascade, eye_pair_cascade
		face_cascade = cv2.CascadeClassifier('./haar-cascade/haarcascade_frontalface_default.xml') # face 
		eye_pair_cascade = cv2.CascadeClassifier('./haar-cascade/haarcascades_haarcascade_mcs_eyepair_big.xml') #eye_pair

	for input_path in tqdm.tqdm(glob.glob(args.input_dir + "/**/*.jpg", recursive=True)):
		save_eye_pair(input_path, os.path.join(args.output_dir, re.sub('^' + args.input_dir, '', input_path)), method=args.method)
		break


# python crop_eye_pair.py --input_dir images/ --output_dir eye_pair_images/ --method mtcnn
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='crop eye pair from face image')
	parser.add_argument('--input_dir', default='', type=str, help='')
	parser.add_argument('--output_dir', default='', type=str, help='')
	parser.add_argument('--method', default='mtcnn', type=str, help='mtcnn or cascade')
	parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
	args = parser.parse_args()

	main(args)