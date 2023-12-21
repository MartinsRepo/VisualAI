import mediapipe as mp
import streamlit as st
import numpy as np
import cv2 as cv

#https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
#nose
NOSE=[ 129, 49, 131, 134, 51, 5, 281, 163, 160, 279, 350, 327, 326, 97, 98] 

ear = [0, 0]

# Define thresholds for drowsiness and distraction detection
drowsy_threshold = 150  # Adjust as needed
distracted_threshold = 15  # Adjust as needed


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



def decode_mediapipe(source, frame, results, face_count, drawing_spec):
	
	dist=[]
	faceXY = []
	image_points = np.empty((0, 2), int)
	
	ih, iw, _ = frame.shape
	
	
	if results.multi_face_landmarks:
		if source == 'image':	
			#Face Landmark Drawing
			
			annotated_image = frame.copy()
			
			for face_landmarks in results.multi_face_landmarks:
				
				f_ar, ip_arr = getImagePoints(face_landmarks, iw, ih)
				faceXY.append(f_ar)		
				image_points = np.append(image_points, ip_arr, axis=0)

				for i in image_points:
					cv.circle(annotated_image,(int(i[0]),int(i[1])),4,(255,0,0),-1)
					
				# Eye and face oval extraction
				rmark, lmark, tilt, faceoval_coords, aspect_ratio_indicator = faceProfileExtractor('image', face_landmarks)
				
				# calculating face oval
				fmark = faceSquareExtractor(faceoval_coords)
				cv.circle(annotated_image,(int(fmark[0]*iw),int(fmark[1]*ih)),6,(255,100,0),-1)
				cv.circle(annotated_image,(int(fmark[2]*iw),int(fmark[3]*ih)),6,(255,100,0),-1)
				
				# draw face oval
				ovalcoords = extractCoordsFromDict(faceoval_coords,iw,ih)
		
				# verify oval coordinates
				pts = np.array(ovalcoords,np.int32)
				cv.polylines(annotated_image, [pts], True, (255,100,100), 5)
				
				


				#print(image_points)
				face_count += 1
				
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
