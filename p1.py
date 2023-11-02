#ip install Selenium
#pip install flask
#pip install opencv-python
#pip install Pillow
#pip install mediapipe 

#Convert videos in readable format
#ffmpeg -i distracted2.mp4 -c:v libx264 -c:a aac -strict -2 adistracted2.mp4


from threading import Timer
import cv2
import os
import sys
import numpy as np
from numpy import arccos, array
from numpy.linalg import norm
import math
import operator
from PIL import Image
import mediapipe as mp
import multiprocessing
import json
from queue_manager import queueConfig
from queue_manager import queueFName
from queue_manager import queueModifiedImage
from queue_manager import queueScenery

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
#drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# 3D model points.
face3Dmodel = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left Mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
    ], dtype=np.float64)
    
#https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 
# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
#nose
NOSE=[ 129, 49, 131, 134, 51, 5, 281, 163, 160, 279, 350, 327, 326, 97, 98] 

UPLOAD_FOLDER_IMG =  os.path.join('images/drvmonpics') 

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

	

# Function to invert normalization of landmarks
#def invert_normalization(x, y, w, h):
#	return int(x * w), int(y * h)


#def fix_aspect_ratio(image, y, y1, required_ratio):
#	new_h = int(image.shape[1] * required_ratio)
#	diff_h = int((new_h - image.shape[0]) / 2)
#	return y - diff_h, y1 + diff_h


#def get_aspect_ratio(region):
#	region_width = region.shape[1]
#	region_height = region.shape[0]
#	region_aspect_ratio = float(region_height) / float(region_width)
#	print("aspect ratio", region_aspect_ratio)
#	return region_aspect_ratio
	
	
# Euclaidean distance 
def euclaideanDistance(point, point1):
	x, y = point
	x1, y1 = point1
	distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
	return distance
	
	
# getting uploaded config data from Flask 
def getQueuingConfig():
	data = None
	try:
		data = queueConfig.get()
	except queueConfig.empty():
		pass
		
	return data
	
	
def checkConfigQueue():
	ret = False
	if not queueConfig.empty():
		ret = True
	return ret


# getting uploaded filename from Flask 	
def getQueuingFName():
	data = None
	try:
		data = queueFName.get_nowait()
	except queueFName.empty():
		pass
		
	return data


def checkFNameQueue():
	ret = False
	if not queueFName.empty():
		ret = True
	return ret


# put modifyied image to the queue
def putQueuingModImage(data):
	while not queueModifiedImage.empty():
		queueModifiedImage.get()
	queueModifiedImage.put(data)
	
		
# put scenery to the queue
def putScenery(data):
	while not queueScenery.empty():
		queueScenery.get()
	queueScenery.put(data)


# Debug file on local disc
def printlandmarks(results):
	for face_landmarks in results:
		with open("Landmarks.txt", "w") as text_file:
			text_file.write("[\n")
			for x in range(sys.getsizeof(face_landmarks)):
				if x%2:
					text_file.write("[%s" % face_landmarks.landmark[x]+"]\n")
			text_file.write("]")


# Returns angle in radians
def calculate_spatial_angles(p1, p2, image_points, nose_end_point2D):
	tmp1=np.array([p1], dtype=object)
	tmp2=np.array([p2], dtype=object)
	vector = np.concatenate((tmp1, tmp2))

	delta_x = image_points[0][0] - nose_end_point2D[0][0][0]
	delta_y = image_points[0][1] - nose_end_point2D[0][0][1]

	azimuth_radians = math.atan2(delta_y, delta_x)
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
	
	return r_eye_marker, l_eye_marker


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


def calculate_nose_direction(landmarks,nose):
	if len(landmarks) != 468:
		raise ValueError("Invalid landmarks. Expected 468 landmarks.")

	# Extract relevant landmarks
	nose_landmark_ids = NOSE
	nose_points = [landmarks[i] for i in nose_landmark_ids]

	# Calculate the center of the nose based on selected landmarks
	#center_x = sum(p.x for p in nose_points) / len(nose_points)
	#center_y = sum(p.y for p in nose_points) / len(nose_points)
	center_x = nose[0]
	center_y = nose[1]
	print(center_x,center_y)
	
	# Calculate angles for each direction
	angles = {
	"up": math.degrees(math.atan2(nose_points[0].y - center_y, nose_points[0].x - center_x)),
	"down": math.degrees(math.atan2(center_y - nose_points[0].y, center_x - nose_points[0].x)),
	"left": math.degrees(math.atan2(nose_points[4].y - center_y, nose_points[4].x - center_x)),
	"right": math.degrees(math.atan2(center_y - nose_points[4].y, center_x - nose_points[4].x)),
	}
	#print(angles)
	

	
	
	#angles = {
	#"up": math.degrees(math.atan2(nose_points[0].y - center_y, nose_points[0].x - center_x)),
	#"down": math.degrees(math.atan2(center_y - nose_points[0].y, center_x - nose_points[0].x)),
	#"left": math.degrees(math.atan2(nose_points[4].y - center_y, nose_points[4].x - center_x)),
	#"right": math.degrees(math.atan2(center_y - nose_points[4].y, center_x - nose_points[4].x)),
	#}

	return angles, center_x, center_y

def draw_nose_direction(image, nose_direction, center_x, center_y):
	line_length = 50  # Length of the vector line
	line_color = (255, 0, 0)  # Green color (you can change this)

	# Calculate the endpoint of the vector
	end_x = int(center_x + line_length * math.cos(math.radians(nose_direction)))
	end_y = int(center_y - line_length * math.sin(math.radians(nose_direction)))

	# Draw the vector line on the image
	cv2.line(image, (int(center_x), int(center_y)), (end_x, end_y), line_color, 2)


    
    
def debugFaceConsolePrint(lme,rme,pts,lmark,dlmark,rmark,drmark,lmouthmatched,rmouthmatched):
	print('Left Mouth Egde:',lme)
	print('Right Mouth Egde:',rme)
	print('Face Oval Numpy Array',pts)
	print('Left Eye Marker',lmark)
	print('Left OpenEyeDistance',dlmark)
	print('Right Eye Marker',rmark)
	print('Right OpenEyeDistance',drmark)
	if lmouthmatched >= 0:
	    print("Left Mouth Endpoint is inside the face oval.")
	else:
	    print("Left Mouth Endpoint is outside the face oval.")

	if rmouthmatched >= 0:
	    print("Right Mouth Endpoint is inside the face oval.")
	else:
	    print("Right Mouth Endpoint is outside the face oval.")

		
def decode_mediapipe(image, results, thresholds, consecframes, counter, imageind):

	scenery = "Not Found"
	
	height, width = image.shape[:2]
	size = image.shape
	annotated_image = image.copy()
	

	#Debug writing
	#printlandmarks(results.multi_face_landmarks)
	
	if results.multi_face_landmarks:
		dist=[]
	
		landmarks = results.multi_face_landmarks[0] #only one face
		#print('###',landmarks)
		
		#mp_drawing.draw_landmarks(annotated_image, landmarks, landmark_drawing_spec=drawing_spec) # draw every match
		
		faceXY = []
		ih, iw, _ = annotated_image.shape
		for id,lm in enumerate(landmarks.landmark):                           # loop over all land marks of one face
			x,y = int(lm.x*iw), int(lm.y*ih)
			# print(lm)
			faceXY.append((x, y))                                           # put all xy points in neat array

		image_points = np.array([
			faceXY[19],     # "nose"
			faceXY[152],    # "chin"
			faceXY[226],    # "left eye left"
			faceXY[446],    # "right eye"
			faceXY[57],     # "left mouth"
			faceXY[287]     # "right mouth"
		], dtype="double")
		
		for i in image_points:
			cv2.circle(annotated_image,(int(i[0]),int(i[1])),4,(255,0,0),-1)
			
		
#############################################

		right_coords = [landmarks.landmark[p] for p in RIGHT_EYE]
		left_coords = [landmarks.landmark[p] for p in LEFT_EYE]
		faceoval_coords = [landmarks.landmark[p] for p in FACE_OVAL]

		rmark, lmark = eyesExtractor(right_coords, left_coords)
		fmark = faceSquareExtractor(faceoval_coords)
		
		cv2.circle(annotated_image,(int(fmark[0]*iw),int(fmark[1]*ih)),6,(255,100,0),-1)
		cv2.circle(annotated_image,(int(fmark[2]*iw),int(fmark[3]*ih)),6,(255,100,0),-1)
		
		ovalcoords = extractCoordsFromDict(faceoval_coords,iw,ih)
		
		# verify oval coordinates
		pts = np.array(ovalcoords,np.int32)
		cv2.polylines(annotated_image, [pts], True, (255,100,100), 5)
		
		## calculate, whether a person looks to the left(right) side or looks in straight direction
		lpoint_inside_oval = cv2.pointPolygonTest(pts, (image_points[4][0], image_points[4][1]), False)
		rpoint_inside_oval = cv2.pointPolygonTest(pts, (image_points[5][0], image_points[5][1]), False)
		
		#nose_direction, cx, cy = calculate_nose_direction(landmarks.landmark,image_points[0])
		#print("Nose Direction (Degrees):")
		#print("Up:", nose_direction["up"])
		#print("Down:", nose_direction["down"])
		#print("Left:", nose_direction["left"])
		#print("Right:", nose_direction["right"])
		# draw vector head position
		#draw_nose_direction(annotated_image, nose_direction["up"], cx, cy)
		#cv2.line(annotated_image, p1, p2, (255, 0, 0), 2)
		
		#calculating nose direction vector
		maxXY = max(faceXY, key=x_element)[0], max(faceXY, key=y_element)[1]
		minXY = min(faceXY, key=x_element)[0], min(faceXY, key=y_element)[1]

		#xcenter = (maxXY[0] + minXY[0]) / 2
		#ycenter = (maxXY[1] + minXY[1]) / 2
		xcenter = int(image_points[0][0])
		ycenter = int(image_points[0][1])
		#print(xcenter,ycenter )
		# faceID, distance, maxXY, minXY
		dist.append((0, (int(((xcenter-width/2)**2+(ycenter-height/2)**2)**.4)), maxXY, minXY)) 
		#print(image_points)
		
		dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
		focal_length = size[1]
		center = (size[1] / 2, size[0] / 2)
		camera_matrix = np.array(
			[[focal_length, 0, center[0]],
			[0, focal_length, center[1]],
			[0, 0, 1]], dtype="double"
		)

		(success, rotation_vector, translation_vector) = cv2.solvePnP(face3Dmodel, image_points,  camera_matrix, dist_coeffs)
		
		(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		
		# draw vector head position
		p1 = (int(image_points[0][0]), int(image_points[0][1]))
		p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
		cv2.line(annotated_image, p1, p2, (255, 0, 0), 2)
		
		noseDirAng = calculate_spatial_angles(p1, p2, image_points, nose_end_point2D)

		print("Azimuth Angle in degrees:", noseDirAng)
		print(success, rotation_vector, translation_vector)
		print(nose_end_point2D, jacobian)
		
		
		# Create scenery marker
		lme=False	# left mouth endpoint
		rme=False	# right mouth endpoint
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
		print('h',facedir_horiz) 
		print(rotation_vector[1],rotation_vector[2])
			
		# face position up-straight-down
		facedir_vert = 99
		if facedir_horiz == -1:
			if int(noseDirAng[0])<0 and rotation_vector[1]>3 and is_between(-0.6, rotation_vector[2], 0.6):
				facedir_vert = 0  # straight
			if int(noseDirAng[0])>0 and is_between(-3, rotation_vector[1], 0) and is_between(-0.6, rotation_vector[2], 0.6):
				facedir_vert = 0  # left			
		elif facedir_horiz==0:
			if int(noseDirAng[0])<0 and is_between(-0.6, rotation_vector[1], 0.6) and is_between(-0.6, rotation_vector[2], 0.6):
				facedir_vert = 0  # straight
			elif int(noseDirAng[0])>0 and is_between(-0.6, rotation_vector[1], 0.6) and is_between(-0.6, rotation_vector[2], 0.6):
				facedir_vert = 1  # up
			elif int(noseDirAng[0])<0 and is_between(-3, rotation_vector[1], 0) and is_between(0, rotation_vector[2], 1.5):
				facedir_vert = -1  # down
			print('#',facedir_vert,int(noseDirAng[0]),is_between(-3, rotation_vector[1], 0), is_between(0, rotation_vector[2], 1.5))
		elif facedir_horiz==1:
			if int(noseDirAng[0])<0 and is_between(-0.6, rotation_vector[1], 0.6) and is_between(-0.6, rotation_vector[2], 0.6):
				facedir_vert = 0  # straight
			if int(noseDirAng[0])<0 and is_between(0.61, rotation_vector[1], 3) and is_between(-0.6, rotation_vector[2], 0.6):
				facedir_vert = 0  # right
		
		
		#if int(noseDirAng[0])<0 and is_between(-0.6, rotation_vector[1], 0.2) and is_between(-0.6, rotation_vector[2], 0.5):
		#	facedir_vert = 0  # straight
		#elif is_between(-0.03, rotation_vector[1], -0.08) and rotation_vector[2] > -0.6 :
		#elif is_between(45, int(noseDirAng[0]), 135):
		#elif noseDirAng[0]>0:
		#	facedir_vert = 1 # up
		#elif  rotation_vector[1] < -2 and is_between(0.21, rotation_vector[2], 1.5):
		#	facedir_vert = -1 #down
		#print('v',facedir_vert) 
			
		# Calculate eye vertikal distances
		rdelta = rmark[3] - rmark[2]
		ldelta = lmark[3] - lmark[2]
		drowsy = False
		if rdelta < 0.03 and ldelta < 0.03:
			drowsy = True
		
		if drowsy:
			if facedir_horiz==0:
				if facedir_vert == 0:
					scenery = "drowsy, face turned into straight direction, looking forward"
				elif facedir_vert == -1:
					scenery = "drowsy, face turned into straight direction, looking downward"
				elif facedir_vert == 1:
					scenery = "drowsy, face turned into straight direction, looking upward"
				elif facedir_vert == 99:
					scenery = "drowsy and looking in straight direction, vertical pose not detectable"
			elif facedir_horiz==-1:
				if facedir_vert == 0:
					scenery = "drowsy, face turned into left direction, looking forward"
				elif facedir_vert == -1:
					scenery = "drowsy, face turned into left direction, looking downward"
				elif facedir_vert == 1:
					scenery = "drowsy, face turned into left direction, looking upward"
				elif facedir_vert == 99:
					scenery = "drowsy, face turned into left direction, vertical pose not detectable"
			elif facedir_horiz==1:
				if facedir_vert == 0:
					scenery = "drowsy, face turned into right direction, looking forward"
				elif facedir_vert == -1:
					scenery = "drowsy, face turned into right direction, looking downward"
				elif facedir_vert == 1:
					scenery = "drowsy, face turned into right direction, looking upward"
				elif facedir_vert == 99:
					scenery = "drowsy and looking in right direction, vertical pose not detectable"
			elif facedir_horiz==99:
				scenery = "drowsy and direction not detectable"
		else:
			if facedir_horiz==0:
				if facedir_vert == 0:
					scenery = "awake, face turned into straight direction, looking forward"
				elif facedir_vert == -1:
					scenery = "awake, face turned into straight direction, looking downward"
				elif facedir_vert == 1:
					scenery = "awake, face turned into straight direction, looking upward"
				elif facedir_vert == 99:
					scenery = "awake, face turned into straight direction, vertical pose not detectable"
			elif facedir_horiz==-1:
				if facedir_vert == 0:
					scenery = "awake, face turned into left direction, looking forward"
				elif facedir_vert == -1:
					scenery = "awake, face turned into left direction, looking downward"
				elif facedir_vert == 1:
					scenery = "awake, face turned into left direction, looking upward"
				elif facedir_vert == 99:
					scenery = "awake, face turned into left direction, vertical pose not detectable"
			elif facedir_horiz==1:
				if facedir_vert == 0:
					scenery = "awake, face turned into right direction, looking forward"
				elif facedir_vert == -1:
					scenery = "awake, face turned into right direction, looking downward"
				elif facedir_vert == 1:
					scenery = "awake, face turned into right direction, looking upward"
				elif facedir_vert == 99:
					scenery = "awake, face turned into right direction, vertical pose not detectable"
			elif facedir_horiz==99:
				scenery = "awake and direction not detectable"


		#debud print - uncomment
		#debugFaceConsolePrint(image_points[4],image_points[5],pts,lmark,ldelta,rmark,rdelta,
		#	lpoint_inside_oval,rpoint_inside_oval)
		
		#if pose == -1:
		#	draw_nose_direction(annotated_image, nose_direction["left"], cx, cy)
		#	print('#')
		#elif pose == 0:
	#		draw_nose_direction(annotated_image, nose_direction["right"], cx, cy)
	#		print('##')
	#	elif pose == 1:
	#		draw_nose_direction(annotated_image, nose_direction["right"], cx, cy)	
	#		print('###')
				
		mp_drawing.draw_landmarks(
			image=annotated_image,
			landmark_list=landmarks,
			connections=mp_face_mesh.FACEMESH_CONTOURS,
			landmark_drawing_spec=None,
			connection_drawing_spec=mp_drawing_styles
			.get_default_face_mesh_contours_style()
		)






#############################################

		

		
			
		dist.sort(key=y_element)
		# print(dist)
		
		# draw square around the detected face
		for i,landmarks in enumerate(results.multi_face_landmarks):
			if i == 0:
				cv2.rectangle(annotated_image,dist[i][2],dist[i][3],(0,255,0),2)
			else:
				cv2.rectangle(annotated_image, dist[i][2], dist[i][3], (0, 0, 255), 2)
		

	return {
		"modified_image": annotated_image,
		"scenery": scenery
	}






def mediapipeprocess():
	drowsiness_threshold 	= 0.84
	awake_threshold 	= 0.85
	distraction_threshold 	= 0.99
	mar_threshold 		= 0.06
	
	consecutive_frames_distraction = 10
	consecutive_frames_drowsiness  = 10
	consecutive_frames_awake       = 10
	consecutive_frames_smiling     = 10
		
	frame_counter 		= 0
	distraction_counter 	= 0
	drowsiness_counter 	= 0
	awake_counter 		= 0
	smiling_counter 	= 0
	last_state 		= None
	
	consecframes = [consecutive_frames_distraction, consecutive_frames_drowsiness, consecutive_frames_awake, consecutive_frames_smiling]
	counter = [frame_counter, distraction_counter, drowsiness_counter, awake_counter, smiling_counter, last_state]
	thresholds = [drowsiness_threshold, awake_threshold, distraction_threshold, mar_threshold]
	fname=""
	
	
	
	
	while True:
		if checkConfigQueue():
			config = getQueuingConfig()
			print("Configuration Settings:", config)
			drowsiness_threshold 	= config["drowsiness"]
			awake_threshold 	= config["awake"]
			distraction_threshold 	= config["distraction"]
			mar_threshold 		= config["smile"]
			
			thresholds = [drowsiness_threshold, awake_threshold, distraction_threshold, mar_threshold]
		
		if checkFNameQueue():
			
			f = getQueuingFName()

			key, fn = list(f.items())[0]
			fname=''.join(fn)
			extension = os.path.splitext(fname)[1]

			print("filename",fname)
			
			#load image
			if extension=='.jpg' or extension=='.jpeg':
			
				with mp_face_mesh.FaceMesh(
					static_image_mode=True,
					max_num_faces=1,
					refine_landmarks=False,
					min_tracking_confidence=0.01,
					min_detection_confidence=0.5) as face_mesh:
					
					#for idx, file in enumerate(IMAGE_FILES):
					image = cv2.imread(UPLOAD_FOLDER_IMG+"/"+fname)
					
					resized_image = cv2.resize(image, (640, 480), interpolation= cv2.INTER_LINEAR) 
					
					# Convert the BGR image to RGB before processing.
					results = face_mesh.process(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

					# calculate face mesh landmarks on the image
					if not results.multi_face_landmarks:
						continue
					
					result = decode_mediapipe(resized_image, results, thresholds, consecframes, counter, True)
				print(result["scenery"])
				# Send scenery to Flask server via a queue 
				putScenery(result["scenery"])	
				# Send the modified image to Flask server via a queue
				putQueuingModImage(result["modified_image"])
					
	

if __name__ == '__main__':
	mediapipeprocess()
	
