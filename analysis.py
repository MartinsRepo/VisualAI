import mediapipe as mp
import streamlit as st
import numpy as np
import torch
import cv2 as cv
import math
from PIL import Image
from io import BytesIO
import base64
import gc
import random
import threading
from facenet_pytorch import MTCNN


stframe = st.empty()
sttext = st.empty()
stdebug = st.empty()
disp = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
		static_image_mode=True,
		max_num_hands=2,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	)


#https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
#nose
NOSE=[ 129, 49, 131, 134, 51, 5, 281, 163, 160, 279, 350, 327, 326, 97, 98] 

#dictionary for 5 faces 
result_dict = {
	'f1': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] },
	'f2': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] },
	'f3': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] },
	'f4': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] },
	'f5': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] } }

# 3D model points.
face3Dmodel = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left Mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
    ], dtype=np.float64)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], # MTCNN thresholds
              factor=0.709,
              post_process=True,
              device=device # If you don't have GPU
        )

ear = [0, 0]

# Define thresholds for drowsiness and distraction detection
drowsy_threshold = 150  # Adjust as needed
distracted_threshold = 15  # Adjust as needed

# Dictionary for optional debug.txt output - please reconfigure to your needs 
debug2text = {"yaw":[],"pitch":[], "roll":[], "nose":[], "tilt":[], "out":[] } 

# Helper functions
def x_element(elem):
	return elem[0]
    
    
def y_element(elem):
	return elem[1]
    
    
def is_between(a, x, b):
	return min(a, b) < x < max(a, b)


# converts <class 'mediapipe.framework.formats.landmark_pb2.NormalizedLandmark'> to dictionary
def convert(NormalizedLandmark):
	res_dict = {}
	xpoints = []
	ypoints = []
	for data_point in NormalizedLandmark:
		xpoints.append(data_point.x)
		ypoints.append(data_point.y)
		
	res_dict["X"]=xpoints
	res_dict["Y"]=ypoints

	return res_dict


# Landmarks: [Left Eye], [Right eye], [nose], [left mouth], [right mouth]
def npAngle(a, b, c):
	ba = a - b
	bc = c - b 

	cosine_angle = np.dot(ba, bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
	angle = np.arccos(cosine_angle)

	return np.degrees(angle)


def debugWrite(imgname,debug2text): 
	with open('debug.txt', 'a') as f:
		f.write("\n\n")
		f.write(imgname + "\n")
		for key, value in debug2text.items():
			f.write(key+": "+value+"\n")


# Returns angle in radians
def calculate_spatial_angles(p1, p2):

	delta_x = p2[0] - p1[0]
	delta_y = p2[1] - p1[1]

	azimuth_radians = math.atan2(delta_x, delta_y)
	azimuth_degrees = math.degrees(azimuth_radians)
	
	return azimuth_degrees, azimuth_radians


def calculate_euler_angles_from_rotation_matrix(R):
	"""
	Calculate Euler angles (yaw, pitch, and roll) from a given rotation matrix.
	Assumes a ZYX rotation order.

	:param R: A 3x3 rotation matrix.
	:return: A tuple of Euler angles (yaw, pitch, roll) in radians.
	"""
	# Ensure R is a numpy array
	R = np.array(R)

	# Calculating Euler angles
	pitch = np.arcsin(-R[2, 0])  # Pitch (θ)
	yaw = np.arctan2(R[1, 0], R[0, 0])  # Yaw (ψ)
	roll = np.arctan2(R[2, 1], R[2, 2])  # Roll (φ)

	return yaw, pitch, roll


# Eyes Extrctor function,
def eyesExtractor(right_eye_coords, left_eye_coords):
	
	d_right_eye_coords=convert(right_eye_coords)
	d_left_eye_coords=convert(left_eye_coords)

	# For RIGHT Eye
	r_max_x = max(d_right_eye_coords["X"])
	r_min_x = min(d_right_eye_coords["X"])
	r_max_y = max(d_right_eye_coords["Y"])
	r_min_y = min(d_right_eye_coords["Y"])
	
	# For LEFT Eye
	l_max_x = max(d_left_eye_coords["X"])
	l_min_x = min(d_left_eye_coords["X"])
	l_max_y = max(d_left_eye_coords["Y"])
	l_min_y = min(d_left_eye_coords["Y"])
	
	r_eye_marker = [r_min_x,r_max_x,r_min_y,r_max_y]
	l_eye_marker = [l_min_x,l_max_x,l_min_y,l_max_y]
	
	# Calculate the Y deviation of the right eye from the left eye
	y_deviation = r_min_y - l_min_y
	
	return r_eye_marker, l_eye_marker, y_deviation
	

# "Eye Aspect Ratio" (EAR) introduced by Soukupová and Čech in their paper 
# "Real-Time Eye Blink Detection Using Facial Landmarks."
def eye_aspect_ratio(eye):
	# Calculate the EAR for a given eye (e.g., eye = [p1, p2, p3, p4, p5, p6])
	# Calculate the distances between eye landmarks
	A = np.sqrt((eye[1].x - eye[5].x) ** 2 + (eye[1].y - eye[5].y) ** 2 + (eye[1].z - eye[5].z) ** 2)
	B = np.sqrt((eye[2].x - eye[4].x) ** 2 + (eye[2].y - eye[4].y) ** 2 + (eye[2].z - eye[4].z) ** 2)
	C = np.sqrt((eye[0].x - eye[3].x) ** 2 + (eye[0].y - eye[3].y) ** 2 + (eye[0].z - eye[3].z) ** 2)

	# Calculate the EAR
	ear = (A + B) / (2.0 * C)
	return ear


# Face Square Extrctor function,
def faceSquareExtractor(faceoval):
	
	d_faceoval_coords=convert(faceoval)
	
	# max/ min values
	max_x = max(d_faceoval_coords["X"])
	min_x = min(d_faceoval_coords["X"])
	max_y = max(d_faceoval_coords["Y"])
	min_y = min(d_faceoval_coords["Y"])
	
	faceoval_marker = [min_x,min_y,max_x,max_y]
		
	return faceoval_marker


### Hand detection
def detect_hands(hands, image):
	
	results = hands.process(image)
	
	if results.multi_hand_landmarks:
		return results.multi_hand_landmarks
	else:
		return None



def checkHandDistraction(height, width, hand_landmarks, threshold):
	if hand_landmarks is None:
		return False  # No hand landmarks detected, not distracted
		
	bottom_right = (0, height - height // 3)
	bottom_left = (width, height - height // 3)

	# Create a NumPy array containing the two coordinates of the crossline
	crossline = np.array([bottom_left, bottom_right], dtype=int)

	landmarks_below_crossline = sum(1 for landmark in hand_landmarks[0].landmark if landmark.y * height > height // 3)

	# Check if the number of landmarks below the crossline exceeds the threshold
	is_below_crossline = landmarks_below_crossline >= threshold

	return is_below_crossline


def extractCoordsFromDict(faceoval_coords,iw,ih):
	
	converted = convert(faceoval_coords)
	
	xcoords=[]
	ycoords=[]
	
	for u,v in converted .items():
		for i in range(len(v)):
			if u=='X':
				xcoords.append(v[i])
			if u=='Y':
				ycoords.append(v[i])
	
	result = []
	for i in range(len(xcoords)):
		x = int(xcoords[i]*iw)
		y = int(ycoords[i]*ih)
		result.append((x, y))

	return result
	

# get 2d canonical face matrix and central image points
def getImagePoints(landmarks, iw, ih):
	faceXY = []
	
	for lm in enumerate(landmarks.landmark):	# loop over all land marks of one face

		# Splitting the string into parts
		parts = lm[1]
		
		# Extracting x and y values
		x = int(parts.x*iw)
		y = int(parts.y*ih)

		faceXY.append((x, y))			# put all xy points in neat array

	image_points = np.array([
		faceXY[19],     # "nose"
		faceXY[152],    # "chin"
		faceXY[226],    # "left eye left"
		faceXY[446],    # "right eye"
		faceXY[57],     # "left mouth"
		faceXY[287]     # "right mouth"
	], dtype="double")
	
	# remember nose position
	nose_2d = faceXY[19]
	
	return faceXY, image_points, nose_2d
	
	
def getAspectRatio(right_eye_landmarks, left_eye_landmarks):
	# Calculate EAR for both eyes
	left_eye_ear = eye_aspect_ratio(left_eye_landmarks)
	right_eye_ear = eye_aspect_ratio(right_eye_landmarks)

	# case of static image
	aspect_ratio_indicator = False
	lee = False
	ree = False
	
	ear[0] = int(left_eye_ear*100)
	ear[1] = int(right_eye_ear*100)
	for i in range(2):
		if ear[i] < drowsy_threshold:
			if i==0: 
				lee = True
			else: 
				ree = True
				
	aspect_ratio_indicator = lee and ree
	
	return aspect_ratio_indicator
	
	
def faceProfileExtractor(source, landmarks):
	# Eye and face oval extraction
	right_eye_landmarks = [landmarks.landmark[p] for p in RIGHT_EYE]
	left_eye_landmarks = [landmarks.landmark[p] for p in LEFT_EYE]
	faceoval_coords = [landmarks.landmark[p] for p in FACE_OVAL]
	
	rmark, lmark, tilt = eyesExtractor(right_eye_landmarks, left_eye_landmarks)
	
	tilt=tilt*100
	
	asr_ind = getAspectRatio(right_eye_landmarks, left_eye_landmarks)
	
	return rmark, lmark, tilt, faceoval_coords, asr_ind


def predFacePose(frame):
	bbox_, prob_, landmarks_ = mtcnn.detect(frame, landmarks=True) # The detection part producing bounding box, probability of the detected face, and the facial landmarks
	angle_R_List = []
	angle_L_List = []
	predLabelList = []
	boxes = []
	
	def checkEye(landmarks):
		angR = npAngle(landmarks[0], landmarks[1], landmarks[2]) # Calculate the right eye angle
		angL = npAngle(landmarks[1], landmarks[0], landmarks[2])# Calculate the left eye angle
		
		angle_R_List.append(angR)
		angle_L_List.append(angL)
		
		if ((int(angR) in range(30, 65)) and (int(angL) in range(30, 65))):
		#if ((int(angR) in range(40, 62)) and (int(angL) in range(32, 59))):
			predLabel='Frontal'
			predLabelList.append(predLabel)
		else: 
			if angR < angL:
				predLabel='RightProfile'
			else:
				predLabel='LeftProfile'
		
		return predLabel
	
	try:
		for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
			if bbox is not None: # To check if we detect a face in the image
				boxes.append(bbox)
				if prob > 0.8: # To check if the detected face has probability more than 80%, to avoid 
					predLabel = checkEye(landmarks)
				else:
					predLabel='NotDetectable'
			else:
				predLabel='NotFace'
			
			predLabelList.append(predLabel)
	except: 
		pass
		
	return boxes, predLabelList


def noseVector(faceXY, image_points, size):
	
	height, width = size[:2]
	p1 = p2 = None
	
	dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
	focal_length = size[1]
	center = (size[1] / 2, size[0] / 2)
	camera_matrix = np.array(
		[[focal_length, 0, center[0]],
		[0, focal_length, center[1]],
		[0, 0, 1]], dtype="double"
	) 
	
	(success, rotation_vector, translation_vector) = cv.solvePnP(face3Dmodel, image_points,  camera_matrix, dist_coeffs)
	
	(nose_end_point2D, _) = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		
	# draw vector head position
	p1 = (int(image_points[0][0]), int(image_points[0][1]))
	p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
	
	# Get rotational matrix
	#rotation_axis = rotation_vector / np.linalg.norm(rotation_vector)
	rotationmatrix = cv.Rodrigues(rotation_vector)[0] # * rotation_axis
	
	# Get angles
	#Yaw (ψ) is the rotation about the Z-axis,
	#Pitch (θ) is the rotation about the Y-axis
	angles,mtxR,mtxQ,Qx,Qy,Qz = cv.RQDecomp3x3(rotationmatrix)
	yaw = angles[0]
	pitch = angles[1]
	roll= angles[2]

	# uncomment for debugging
	#print("Yaw (ψ):", yaw)
	#print("Pitch (θ):", pitch)
	#print("Roll (φ):", roll)

	rotangles = (yaw,pitch,roll)
	
	return p1, p2, rotation_vector, rotangles 


def scenery_handdetection(height, width, hand_landmarks):
	
	# hands
	str_hands=''
	is_below=False
	
	# 15 is the threshold and can be set dynamical tbd
	is_below = checkHandDistraction(height, width, hand_landmarks,15)

	if is_below:
	    str_hands="Hands below the face."
	else:
	    str_hands="Hands near the face."

	return str_hands, is_below


def create_scenerymarker(LabelList, lpoint_inside_oval, rpoint_inside_oval, aspect_ratio_indicator, rotangles, noseDirAng, rmark, lmark, tilt, is_below):
	
	lme=False	# left mouth endpoint
	rme=False	# right mouth endpoint
	
	# head position
	if lpoint_inside_oval >= 0:
		lme = True
	if rpoint_inside_oval >= 0:
		rme = True
	
	### face position left - straight - right
	facedir_horiz = 99
	if lme and rme:  # always frontal, override LabelList
		if is_between(-90, noseDirAng[0], -40):
			facedir_horiz = 1 			# right
		elif  is_between(40, noseDirAng[0], 90):
			facedir_horiz = -1			# left
		else:
			facedir_horiz = 0  			# straight
	
	elif LabelList == "LeftProfile" or (lme and not rme):
		facedir_horiz = -1  				# left
	elif LabelList == "RightProfile"or (not lme and rme):
		facedir_horiz = 1  				# right
	
	# face position up-straight-down
	facedir_vert = 99
	
	# detect vertical position
	if facedir_horiz == 0:
		if rotangles[0]<0: # negativ Yaw for straight/down
			if is_between(0, abs(rotangles[1]), 10):	# Yaw range
				facedir_vert = 0		# forward
			if abs(rotangles[1]) > 10:
				facedir_vert= -1		# down
		else:
			if is_between(0, abs(rotangles[1]), 25): 	# Yaw range
				facedir_vert = 1		# up
			else:
				facedir_vert= -1		# down

	elif facedir_horiz == -1: #profileleft
		if rotangles[0]<0: 					# negativ Yaw for straight/down
				facedir_vert = -1 		#up	
		else :
			if is_between(110, abs(noseDirAng[0]), 180) :		
				facedir_vert = 1 		#up	
			elif abs(noseDirAng[0])<110:
				facedir_vert = 0
	elif facedir_horiz == 1: #profileright
		if rotangles[0]<0: 					# negativ Yaw for straight
			if is_between(10, abs(rotangles[1]), 90):
				facedir_vert = 0		# forward
			elif abs(rotangles[1]) < 10:
				facedir_vert = -1		# down
		else:
			if is_between(0, abs(rotangles[1]), 45):
				facedir_vert = -1 		#down
			else:
				facedir_vert = 1 		#up
			
	# Calculate eye vertikal distances
	rdelta = rmark[3] - rmark[2]
	ldelta = lmark[3] - lmark[2]
	drowsy = False
	distracted = False
	awake = False
	
	if facedir_vert == -1 and is_below:
		distracted = True
	elif rdelta < 0.03 and ldelta < 0.03 and aspect_ratio_indicator:
		drowsy = True
	else:
		awake = True
	
	return facedir_horiz, facedir_vert, awake, drowsy, distracted


def scenery_description(awake, drowsy, distracted, str_hands,  facedir_horiz, facedir_vert):
	scenery = ''
	if drowsy:
		if facedir_horiz==0:
			if facedir_vert == 0:
				scenery = "Person drowsy, face turned into straight direction, looking forward. "+str_hands
			elif facedir_vert == -1:
				scenery = "Person drowsy, face turned into straight direction, looking downward. "+str_hands
			elif facedir_vert == 1:
				scenery = "Person drowsy, face turned into straight direction, looking upward. "+str_hands
			elif facedir_vert == 99:
				scenery = "Person drowsy and looking in straight direction, vertical pose not detectable. "+str_hands
		elif facedir_horiz==-1:
			if facedir_vert == 0:
				scenery = "Person drowsy, face turned into left direction, looking forward. "+str_hands
			elif facedir_vert == -1:
				scenery = "Person drowsy, face turned into left direction, looking downward. "+str_hands
			elif facedir_vert == 1:
				scenery = "Person drowsy, face turned into left direction, looking upward. "+str_hands
			elif facedir_vert == 99:
				scenery = "Person drowsy, face turned into left direction, vertical pose not detectable. "+str_hands
		elif facedir_horiz==1:
			if facedir_vert == 0:
				scenery = "Person drowsy, face turned into right direction, looking forward. "+str_hands
			elif facedir_vert == -1:
				scenery = "Person drowsy, face turned into right direction, looking downward. "+str_hands
			elif facedir_vert == 1:
				scenery = "Person drowsy, face turned into right direction, looking upward. "+str_hands
			elif facedir_vert == 99:
				scenery = "Person drowsy and looking in right direction, vertical pose not detectable. "+str_hands
		elif facedir_horiz==99:
			scenery = "Person drowsy and direction not detectable. "+str_hands
	
	elif distracted:
		if facedir_horiz==0:
			scenery = "Person distracted, face turned into straight direction, looking downward. "+str_hands
		elif facedir_horiz==-1:
			scenery = "Person distracted, face turned into left direction, looking downward. "+str_hands
		elif facedir_horiz==1:
			scenery = "Person distracted, face turned into right direction, looking downward. "+str_hands
	
	elif awake:
		if facedir_horiz==0:
			if facedir_vert == 0:
				scenery = "Person awake, face turned into straight direction, looking forward. "+str_hands
			elif facedir_vert == -1:
				scenery = "Person awake, face turned into straight direction, looking downward. "+str_hands
			elif facedir_vert == 1:
				scenery = "Person awake, face turned into straight direction, looking upward. "+str_hands
			elif facedir_vert == 99:
				scenery = "Person awake, face turned into straight direction, vertical pose not detectable. "+str_hands
		elif facedir_horiz==-1:
			if facedir_vert == 0:
				scenery = "Person awake, face turned into left direction, looking forward. "+str_hands
			elif facedir_vert == -1:
				scenery = "Person awake, face turned into left direction, looking downward. "+str_hands
			elif facedir_vert == 1:
				scenery = "Person awake, face turned into left direction, looking upward. "+str_hands
			elif facedir_vert == 99:
				scenery = "Person awake, face turned into left direction, vertical pose not detectable. "+str_hands
		elif facedir_horiz==1:
			if facedir_vert == 0:
				scenery = "Person awake, face turned into right direction, looking forward. "+str_hands
			elif facedir_vert == -1:
				scenery = "Person awake, face turned into right direction, looking downward. "+str_hands
			elif facedir_vert == 1:
				scenery = "Person awake, face turned into right direction, looking upward. "+str_hands
			elif facedir_vert == 99:
				scenery = "Person awake, face turned into right direction, vertical pose not detectable. "+str_hands
		elif facedir_horiz==99:
			scenery = "Person awake and direction not detectable. "+str_hands
	else:
		scenery = "Person state not dectable. "
	
	return scenery


# Resize Images to fit Container
@st.cache_data ()
# Get Image Dimensions
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    dim = None
    # grab the image size
    (h,w) = image.shape[:2]

    if width is None and height is None:
        return image
    # calculate the ratio of the height and construct the
    # dimensions
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv.resize(image,dim,interpolation=inter)

    return resized

#def SortpredLabelList()    

def decode_image_mediapipe(frame, imgfilename, results, face_count, left_placeholder, right_placeholder, debug_mode):
	scenery = [None] * 10 # max 10 faces
	LabelList = []
	
	drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
	
	if results.multi_face_landmarks:
		
		#Face Landmark Drawing
		annotated_image = image_resize(frame, width=313, height=438)
		ih, iw, _ = annotated_image.shape
		
		boxes, predLabelList = predFacePose(annotated_image)
		blen = len(boxes)
	
		#draw rectangle, uncomment for debugging
		#cv.rectangle(annotated_image,(int(boxes[0][0]),int(boxes[0][1])),(int(boxes[0][2]),int(boxes[0][3])),(255,0,0),2)
		
		for face_landmarks in results.multi_face_landmarks:
			faceXY = []
			image_points = np.empty((0, 2), int)
			f_arr, ip_arr, _ = getImagePoints(face_landmarks, iw, ih)
			
			faceXY.append(f_arr)		
			image_points = np.append(image_points, ip_arr, axis=0)
			
			for i in image_points:
				cv.circle(annotated_image,(int(i[0]),int(i[1])),2,(255,0,0),-1)
			
			# Make faceprediction List
			def solve(bl, tr, p) :
				if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
					return True
				else :
					return False
			
			i=0
			for i in range(blen):
				bottom_left = (boxes[i][0],boxes[i][1])
				top_right = (boxes[i][2],boxes[i][3])
				nosepoint = (ip_arr[0][0], ip_arr[0][1])
				
				if solve(bottom_left, top_right, nosepoint):
					break

			LabelList = predLabelList[i]
							
			# Eye and face oval extraction
			rmark, lmark, tilt, faceoval_coords, aspect_ratio_indicator = faceProfileExtractor('image', face_landmarks)
			
			# calculating face oval
			fmark = faceSquareExtractor(faceoval_coords)
			cv.circle(annotated_image,(int(fmark[0]*iw),int(fmark[1]*ih)),2,(255,100,0),-1)
			cv.circle(annotated_image,(int(fmark[2]*iw),int(fmark[3]*ih)),2,(255,100,0),-1)
			
			# draw face oval
			ovalcoords = extractCoordsFromDict(faceoval_coords,iw,ih)
	
			# verify oval coordinates
			pts = np.array(ovalcoords,np.int32)
			cv.polylines(annotated_image, [pts], True, (255,100,100), 2)
			
			## calculate, whether a person looks to the left(right) side or looks in straight direction
			lpoint_inside_oval = cv.pointPolygonTest(pts, (image_points[4][0], image_points[4][1]), False)
			rpoint_inside_oval = cv.pointPolygonTest(pts, (image_points[5][0], image_points[5][1]), False)
			
			# nose vector 2d representation
			p1, p2, rotation_vector, rotangles = noseVector(faceXY, image_points, frame.shape)
			cv.line(annotated_image, p1, p2, (238, 255, 0), 3)

			noseDirAng = calculate_spatial_angles(p1, p2)

			# calculate and draw hand positions
			hand_landmarks = detect_hands(hands, annotated_image)
			
			is_below = None
			str_hands =''
			if hand_landmarks is not None:
				
				for hand_landmark in hand_landmarks:
					# Calculate the bounding box for the hands
					x_max = 0
					y_max = 0
					x_min = iw
					y_min = ih
					#mp_drawing.draw_landmarks(annotated_image, hand_landmark, mp_hands.HAND_CONNECTIONS)
					for lm in hand_landmark.landmark:
						x, y = int(lm.x * iw), int(lm.y * ih)
						if x > x_max:
							x_max = x
						if y > y_max:
							y_max = y
						if x < x_min:
							x_min = x
						if y < y_min:
							y_min = y
					
					# Draw the bounding box		
					cv.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
					#print(x_min, y_min,x_max, y_max)

				str_hands, is_below = scenery_handdetection(ih, iw, hand_landmarks)
				
			# get scenery markers
			awake = drowsy = distracted = False
			facedir_horiz, facedir_vert, awake, drowsy, distracted = create_scenerymarker(LabelList, lpoint_inside_oval, rpoint_inside_oval, aspect_ratio_indicator, rotangles, noseDirAng, rmark, lmark, tilt, is_below)
			
			# describe the scenery by text
			scene = scenery_description(awake, drowsy, distracted, str_hands,  facedir_horiz, facedir_vert)
			scenery[face_count] = ('Face '+ str(face_count+1) + ': ' + scene)
			
			# writing face number to image
			if int(fmark[1]*ih) > 0:
				cv.putText(annotated_image,str(face_count+1), (int(fmark[0]*iw), int(fmark[1]*ih)), cv.FONT_HERSHEY_PLAIN, 2, (205,108,0),3)
			else:
				cv.putText(annotated_image,str(face_count+1), (int(fmark[2]*iw), int(fmark[3]*ih)), cv.FONT_HERSHEY_PLAIN, 2, (255,108,0),3)

			if debug_mode == 'On':
				debug2text = {'yaw': str(rotangles[0]), 'pitch': str(rotangles[1]), 'roll': str(rotangles[2]), 'nose':str(noseDirAng),'tilt':str(tilt),'out':scenery[face_count]}
				
				debugWrite(imgfilename, debug2text)	
			
			
			face_count += 1
			faceXY = None
			image_points = None
			
			# for display purpose collect results
			key = 'f'+str(face_count)
			result_dict[key] = {'FacedirH': str(facedir_horiz), 'FacedirV': str(facedir_vert), 'Drowsy': str(drowsy), 'Distracted': str(distracted), 
						'RotMatrix': str(np.round(rotation_vector, 2)), 'Tilt': str(np.round(tilt, 2)), 
						'NoseVec': str(np.round(noseDirAng, 2))}
			 
		
		left_placeholder.image(annotated_image, use_column_width=True)

		right_placeholder.markdown("""
				<style>
					.spacer {
						margin-top: 100px;  /* Adjust the size as needed */
					}
				</style>
				<div class="spacer"></div>
			""", unsafe_allow_html=True)
		for i in range(face_count):
			right_placeholder.text_area("Detected Scenery", value=scenery[i], height=100)
		
		if debug_mode == 'On':
			for k in result_dict.keys():
				if result_dict[k]['FacedirH'] :
					right_placeholder.text_area("Debug Window:", value=str(k)+': '+str(result_dict[k]), height=50, key=str(random.random()))
		
		# cleanup
		st.cache_data.clear()
		st.cache_resource.clear()


#Periodically output of the scenery text
def background_task():
	global disp
	disp = True		
	threading.Timer(1, background_task).start()


def decode_video_mediapipe(video, max_faces, detection_confidence, tracking_confidence, debug_mode):
	global stframe, sttext, disp

	result_dict = {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] };
	drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
	
	video_column, text_column = st.columns([2, 1])
	
	with mp.solutions.face_mesh.FaceMesh(
		max_num_faces=max_faces,
		min_detection_confidence=detection_confidence,
		min_tracking_confidence=tracking_confidence

	) as face_mesh:
		face_count = 0
		
		timerthread = threading.Timer(1, background_task)
		timerthread.daemon = True
		timerthread.start()
		
		while video.isOpened():	
			ret, frame = video.read()
			if not ret:
				continue

			results = face_mesh.process(frame)
			frame.flags.writeable = True
			
			ih, iw, _ = frame.shape
			
			if ret:
				boxes, predLabelList = predFacePose(frame)
				blen = len(boxes)
			else:
				boxes = predLabelList = None
				blen = 0
			
			face_count = 0
			
			if results.multi_face_landmarks:
				scenery = [None] * 5 # max 5 faces

				#Face Landmark Drawing
				for face_landmarks in results.multi_face_landmarks:
					
					faceXY = []
					image_points = np.empty((0, 2), int)
					f_arr, ip_arr, _ = getImagePoints(face_landmarks, iw, ih)

					faceXY.append(f_arr)		
					image_points = np.append(image_points, ip_arr, axis=0)
					
					for i in image_points:
						cv.circle(frame,(int(i[0]),int(i[1])),3,(255,0,0),-1)
						
					# Make faceprediction List
					def solve(bl, tr, p) :
						if (p[0] > bl[0] and p[0] < tr[0] and p[1] > bl[1] and p[1] < tr[1]) :
							return True
						else :
							return False
					
					i=0
					if blen>0:
						for i in range(blen):
							bottom_left = (boxes[i][0],boxes[i][1])
							top_right = (boxes[i][2],boxes[i][3])
							nosepoint = (ip_arr[0][0], ip_arr[0][1])
							
							if solve(bottom_left, top_right, nosepoint):
								break

						LabelList = predLabelList[i]
					else:
						LabelList = "No frame"
						
					# Eye and face oval extraction
					rmark, lmark, tilt, faceoval_coords, aspect_ratio_indicator = faceProfileExtractor('image', face_landmarks)
			
					# calculating face oval
					fmark = faceSquareExtractor(faceoval_coords)
					cv.circle(frame,(int(fmark[0]*iw),int(fmark[1]*ih)),2,(255,100,0),-1)
					cv.circle(frame,(int(fmark[2]*iw),int(fmark[3]*ih)),2,(255,100,0),-1)
				
					
					# draw face oval
					ovalcoords = extractCoordsFromDict(faceoval_coords,iw,ih)
			
					# verify oval coordinates
					pts = np.array(ovalcoords,np.int32)
					cv.polylines(frame, [pts], True, (255,100,100), 3)
				
					## calculate, whether a person looks to the left(right) side or looks in straight direction
					lpoint_inside_oval = cv.pointPolygonTest(pts, (image_points[4][0], image_points[4][1]), False)
					rpoint_inside_oval = cv.pointPolygonTest(pts, (image_points[5][0], image_points[5][1]), False)
					
					# nose vector 2d representation
					p1, p2, rotation_vector, rotangles = noseVector(faceXY, image_points, frame.shape)
					cv.line(frame, p1, p2, (238, 255, 0), 4)

					noseDirAng = calculate_spatial_angles(p1, p2)

					# calculate and draw hand positions
					hand_landmarks = detect_hands(hands, frame)
					
					is_below = None
					str_hands =''
					if hand_landmarks is not None:
						
						for hand_landmark in hand_landmarks:
							# Calculate the bounding box for the hands
							x_max = 0
							y_max = 0
							x_min = iw
							y_min = ih
							#mp_drawing.draw_landmarks(annotated_image, hand_landmark, mp_hands.HAND_CONNECTIONS)
							for lm in hand_landmark.landmark:
								x, y = int(lm.x * iw), int(lm.y * ih)
								if x > x_max:
									x_max = x
								if y > y_max:
									y_max = y
								if x < x_min:
									x_min = x
								if y < y_min:
									y_min = y
							
							# Draw the bounding box		
							cv.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
							#print(x_min, y_min,x_max, y_max)

						str_hands, is_below = scenery_handdetection(ih, iw, hand_landmarks)

					#str_hands =''
					#is_below = None
					#if hand_landmarks is not None:
					#	for hand_landmark in hand_landmarks:
					#		mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
					#	str_hands, is_below = scenery_handdetection(ih, iw, hand_landmarks)
				
					# get scenery markers
					awake = drowsy = distracted = False
					facedir_horiz, facedir_vert, awake, drowsy, distracted = create_scenerymarker(LabelList, lpoint_inside_oval, rpoint_inside_oval, aspect_ratio_indicator, rotangles, noseDirAng, rmark, lmark, tilt, is_below)
			
					# describe the scenery by text
					scene = scenery_description(awake, drowsy, distracted, str_hands,  facedir_horiz, facedir_vert)
					
					#facedir_horiz, facedir_vert, drowsy, distracted = create_scenerymarker(LabelList, lpoint_inside_oval, rpoint_inside_oval, aspect_ratio_indicator, rotation_vector, noseDirAng, rmark, lmark, tilt, is_below)
					
					# describe the scenery by text
					#scene = scenery_description(drowsy, distracted, str_hands,  facedir_horiz, facedir_vert)
					
					scenery.append('Face ' +  str(face_count+1) + ': ' + scene+'\n')
					
					# writing face number to image
					if int(fmark[1]*ih) > 0:
						cv.putText(frame, str(face_count+1), (int(fmark[0]*iw), int(fmark[1]*ih)), cv.FONT_HERSHEY_PLAIN, 3, (208,32,144),3)
					else:
						cv.putText(frame, str(face_count+1), (int(fmark[2]*iw), int(fmark[3]*ih)), cv.FONT_HERSHEY_PLAIN, 3, (208,32,144),3)
						
					face_count += 1
					faceXY = None
					image_points = None
					hand_landmarks = None
					
					# for display purpose collect results
					key = 'f'+str(face_count)
					result_dict[key] = {'FacedirH': str(facedir_horiz), 'FacedirV': str(facedir_vert), 'Drowsy': str(drowsy), 'Distracted': str(distracted), 
								'RotMatrix': str(np.round(rotation_vector, 2)), 'Tilt': str(np.round(tilt, 2)), 
								'NoseVec': str(np.round(noseDirAng, 2))}
				
								
				frame = cv.resize(frame,(0,0), fx=0.4, fy=0.4)
				frame = image_resize(image=frame, width=640)
				
				with video_column:
					stframe.image(frame,channels='BGR', use_column_width=False)

				if disp: 
					out = ''
					for scen in scenery:
						if scen is not None:
							out = out + scen + '\n'

					sttext.markdown("Scenery:\n\n"+out)
					
					out = ''
					if debug_mode == 'On':
						for k in result_dict.keys():
							if bool(result_dict.get(k)):
								if result_dict[k]['FacedirH'] :
									out = out+str(k)+': '+str(result_dict[k]) + '\n\n'
						stdebug.markdown("Debug Window:\n\n"+out)
					disp = False
				
					
