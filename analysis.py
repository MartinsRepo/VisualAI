import mediapipe as mp
import streamlit as st
import numpy as np
import cv2 as cv
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

#https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
#nose
NOSE=[ 129, 49, 131, 134, 51, 5, 281, 163, 160, 279, 350, 327, 326, 97, 98] 


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


# Returns angle in radians
def calculate_spatial_angles(p1, p2, image_points, nose_end_point2D):

	#delta_x = abs(p1[0] - p2[0])
	#delta_y = abs(p1[1] - p2[1])
	delta_x = p2[0] - p1[0]
	delta_y = p2[1] - p1[1]

	azimuth_radians = math.atan2(delta_x, delta_y)
	azimuth_degrees = math.degrees(azimuth_radians)
	
	return azimuth_degrees, azimuth_radians


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
def detect_hands(image):
	
	hands = mp_hands.Hands(
		static_image_mode=True,
		max_num_hands=2,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	)
		
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
	distance=[]
	
	#calculating nose direction vector
	maxXY = max(faceXY, key=x_element)[0], max(faceXY, key=y_element)[1]
	minXY = min(faceXY, key=x_element)[0], min(faceXY, key=y_element)[1]
	
	xcenter = int(image_points[0][0])
	ycenter = int(image_points[0][1])
	
	distance.append((0, (int(((xcenter-width/2)**2+(ycenter-height/2)**2)**.4)), maxXY[0], minXY[0]))
	
	
	dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
	focal_length = size[1]
	center = (size[1] / 2, size[0] / 2)
	camera_matrix = np.array(
		[[focal_length, 0, center[0]],
		[0, focal_length, center[1]],
		[0, 0, 1]], dtype="double"
	) 
	print(distance)
	(success, rotation_vector, translation_vector) = cv.solvePnP(face3Dmodel, image_points,  camera_matrix, dist_coeffs)
	
	(nose_end_point2D, jacobian) = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		
	# draw vector head position
	p1 = (int(image_points[0][0]), int(image_points[0][1]))
	p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

	return p1, p2, distance, nose_end_point2D


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

	return str_hands


def decode_mediapipe(source, frame, results, face_count, drawing_spec):
	
	dist=[]
	
	ih, iw, _ = frame.shape
	
	
	if results.multi_face_landmarks:
		if source == 'image':	
			
			#Face Landmark Drawing
			annotated_image = frame.copy()
			
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
				cv.polylines(annotated_image, [pts], True, (255,100,100), 1)
				
				## calculate, whether a person looks to the left(right) side or looks in straight direction
				lpoint_inside_oval = cv.pointPolygonTest(pts, (image_points[4][0], image_points[4][1]), False)
				rpoint_inside_oval = cv.pointPolygonTest(pts, (image_points[5][0], image_points[5][1]), False)
				
				# nose vector 2d representation
				p1, p2, dist, nose_end_point2D = noseVector(faceXY, image_points, frame.shape)
				cv.line(annotated_image, p1, p2, (255, 0, 0), 1)

				noseDirAng = calculate_spatial_angles(p1, p2, image_points, nose_end_point2D)

				# calculate and drwa hand positions
				hand_landmarks = detect_hands(annotated_image)
				
				str_hands =''
				if hand_landmarks is not None:
					for hand_landmark in hand_landmarks:
						mp_drawing.draw_landmarks(annotated_image, hand_landmark, mp_hands.HAND_CONNECTIONS)
					str_hands = scenery_handdetection(ih, iw, hand_landmarks)
				
				
				
				
				
				mp_drawing.draw_landmarks(
					image=annotated_image,
					landmark_list=face_landmarks,
					connections=mp_face_mesh.FACEMESH_CONTOURS,
					landmark_drawing_spec=None,
					connection_drawing_spec=mp_drawing_styles
					.get_default_face_mesh_contours_style()
				)
				
				face_count += 1
				faceXY = None
				image_points = None
				
				#mp.solutions.drawing_utils.draw_landmarks(
				#	image=frame,
				#	landmark_list=face_landmarks,
				#	connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
				#	landmark_drawing_spec=drawing_spec
				#)

			st.image(annotated_image, use_column_width=True)
			
		elif source == 'video':

			#Face Landmark Drawing
			for face_landmarks in results.multi_face_landmarks:
				face_count += 1

			mp.solutions.drawing_utils.draw_landmarks(
				image=frame,
				landmark_list=face_landmarks,
				connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
				landmark_drawing_spec=drawing_spec,
				connection_drawing_spec=drawing_spec
			)
			
	return face_count
