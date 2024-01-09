import mediapipe as mp
import streamlit as st
import numpy as np
import cv2 as cv
import math
from PIL import Image
from io import BytesIO
import base64
import gc
import random
import threading


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


ear = [0, 0]

# Define thresholds for drowsiness and distraction detection
drowsy_threshold = 150  # Adjust as needed
distracted_threshold = 15  # Adjust as needed

# Dictionary for optional debug.txt output - please reconfigure to your needs
debug2text = {"p1":[],"p2":[], "NoseVec Angle":[],"Rotation Vector":[],"LeftEye in Oval":[],"RightEye in Oval":[],"Tilt":[] } 


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
	

# get canonical face matrix and central image points
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

	return faceXY, image_points
	
	
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
	rotation_axis = rotation_vector / np.linalg.norm(rotation_vector)
	rotationmatrix = cv.Rodrigues(rotation_vector)[0] * rotation_axis
	
	# Get angles
	#Yaw (ψ) is the rotation about the Z-axis,
	#Pitch (θ) is the rotation about the Y-axis, and
	#Roll (φ) is the rotation about the X-axis.
	yaw, pitch, roll = calculate_euler_angles_from_rotation_matrix(rotationmatrix)

	print("Yaw (ψ):", yaw)
	print("Pitch (θ):", pitch)
	print("Roll (φ):", roll)

	rotangles = (yaw,pitch, roll)
	
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


def create_scenerymarker(lpoint_inside_oval, rpoint_inside_oval, aspect_ratio_indicator, rotation_vector, noseDirAng, rmark, lmark, tilt, is_below):
	
	lme=False	# left mouth endpoint
	rme=False	# right mouth endpoint
	
	# head position
	if lpoint_inside_oval >= 0:
		lme = True
	else:
	   	lme = False
	rme = True
	if rpoint_inside_oval >= 0:
		rme = True
	else:
		rme = False
		
	# face position left-straight-right
	facedir_horiz = 99
	if lme and rme:
		facedir_horiz = 0  # straight
	elif lme and not rme:
		facedir_horiz = -1 # left
	elif not lme and rme:
		facedir_horiz = 1  # right
	
	# face position up-straight-down
	facedir_vert = 99
	
	# detect vertical position
	if facedir_horiz == 0:
		#print( noseDirAng[0], rotation_vector[0], rotation_vector[1])
		if is_between(-45, noseDirAng[0], 45) and is_between(-3.5, rotation_vector[0], -2) and is_between(-0.25, rotation_vector[1], 0.25):
			facedir_vert = 0
		elif is_between(-10, noseDirAng[0], 10) and  is_between(-2.49, rotation_vector[0], 0):
			facedir_vert= -1
		elif is_between(-180, noseDirAng[0], -160) and is_between(-0.6, rotation_vector[1], 0):
			facedir_vert = 1

	elif facedir_horiz == -1:
		if is_between(60, noseDirAng[0], 170) :
			if is_between(-0.6, rotation_vector[0], 0.6):
				facedir_vert = 0
			elif rotation_vector[0]<0.6 or rotation_vector[0]>0.6 and tilt>=2:
				facedir_vert = 1	
		elif is_between(0, noseDirAng[0], 59) and tilt < -2:
			facedir_vert = -1	
	elif facedir_horiz == 1:
		if is_between(-170, noseDirAng[0], -60) :
			if noseDirAng[0] >- 120 and is_between(-2.5, tilt, 2.5):
				facedir_vert = 0
			elif is_between(-0.8, rotation_vector[0], -0.2):
					facedir_vert = 1
		elif is_between(-59, noseDirAng[0], 0) and tilt>3:
			facedir_vert = -1
	
	# Calculate eye vertikal distances
	rdelta = rmark[3] - rmark[2]
	ldelta = lmark[3] - lmark[2]
	drowsy = False
	distracted = False
	
	if facedir_vert == -1 and is_below:
		distracted = True
	elif rdelta < 0.03 and ldelta < 0.03 and aspect_ratio_indicator:
		drowsy = True
	
	
	return facedir_horiz, facedir_vert, drowsy, distracted


def scenery_description(drowsy, distracted, str_hands,  facedir_horiz, facedir_vert):
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
	
	else:
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
    

def decode_image_mediapipe(frame, imgfilename, results, face_count, left_placeholder, right_placeholder, debug_mode):
	dist=[]
	scenery = [None] * 10 # max 10 faces
	
	drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)
	
	if results.multi_face_landmarks:
		
		#Face Landmark Drawing
		annotated_image = image_resize(frame, width=313, height=438)
		ih, iw, _ = annotated_image.shape
		
		for face_landmarks in results.multi_face_landmarks:
			faceXY = []
			image_points = np.empty((0, 2), int)
			f_arr, ip_arr = getImagePoints(face_landmarks, iw, ih)

			faceXY.append(f_arr)		
			image_points = np.append(image_points, ip_arr, axis=0)
			
			for i in image_points:
				cv.circle(annotated_image,(int(i[0]),int(i[1])),2,(255,0,0),-1)
				
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
			#cv.line(annotated_image, p1, p2, (38, 128, 15), 2)
			cv.line(annotated_image, p1, p2, (238, 255, 0), 3)

			noseDirAng = calculate_spatial_angles(p1, p2)

			# calculate and draw hand positions
			hand_landmarks = detect_hands(hands, annotated_image)
			
			str_hands =''
			is_below = None
			if hand_landmarks is not None:
				for hand_landmark in hand_landmarks:
					mp_drawing.draw_landmarks(annotated_image, hand_landmark, mp_hands.HAND_CONNECTIONS)
				str_hands, is_below = scenery_handdetection(ih, iw, hand_landmarks)
			
			
			# get scenery markers
			facedir_horiz, facedir_vert, drowsy, distracted = create_scenerymarker(lpoint_inside_oval, rpoint_inside_oval, aspect_ratio_indicator, rotation_vector, noseDirAng, rmark, lmark, tilt, is_below)
			
			# describe the scenery by text
			scene = scenery_description(drowsy, distracted, str_hands,  facedir_horiz, facedir_vert)
			scenery[face_count] = ('Face '+ str(face_count+1) + ': ' + scene)
			
			# writing face number to image
			if int(fmark[1]*ih) > 0:
				cv.putText(annotated_image,str(face_count+1), (int(fmark[0]*iw), int(fmark[1]*ih)), cv.FONT_HERSHEY_PLAIN, 2, (205,108,0),3)
			else:
				cv.putText(annotated_image,str(face_count+1), (int(fmark[2]*iw), int(fmark[3]*ih)), cv.FONT_HERSHEY_PLAIN, 2, (255,108,0),3)

			if debug_mode == 'On':
				debug2text = {'p1': str(p1), 'p2': str(p2), 'NoseVec Angle': str(np.round(noseDirAng, 2)), 'Rotation Vector': str(rotation_vector), 'LeftEye in Ova': str(lpoint_inside_oval), 
							'RightEye in Oval': str(lpoint_inside_oval), 'Tilt': str(np.round(tilt, 2))}
				
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
			face_count = 0
			
			if results.multi_face_landmarks:
				scenery = [None] * 5 # max 5 faces

				#Face Landmark Drawing
				for face_landmarks in results.multi_face_landmarks:
					
					faceXY = []
					image_points = np.empty((0, 2), int)
					f_arr, ip_arr = getImagePoints(face_landmarks, iw, ih)

					faceXY.append(f_arr)		
					image_points = np.append(image_points, ip_arr, axis=0)
					
					for i in image_points:
						cv.circle(frame,(int(i[0]),int(i[1])),3,(255,0,0),-1)
						
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

					str_hands =''
					is_below = None
					if hand_landmarks is not None:
						for hand_landmark in hand_landmarks:
							mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
						str_hands, is_below = scenery_handdetection(ih, iw, hand_landmarks)
				
					# get scenery markers
					facedir_horiz, facedir_vert, drowsy, distracted = create_scenerymarker(lpoint_inside_oval, rpoint_inside_oval, aspect_ratio_indicator, rotation_vector, noseDirAng, rmark, lmark, tilt, is_below)
					
					# describe the scenery by text
					scene = scenery_description(drowsy, distracted, str_hands,  facedir_horiz, facedir_vert)
					
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
				
					
