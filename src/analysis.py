import mediapipe as mp
import streamlit as st
import numpy as np
import cv2, math
from PIL import Image
from io import BytesIO
import base64
import gc
import random
import threading

import calculations


stframe = st.empty()
sttext = st.empty()
stdebug = st.empty()
disp = False

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

calc = calculations.CalcFunctions()

hands = calc.mp_hands.Hands(
		static_image_mode=True,
		max_num_hands=2,
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5
	)

#dictionary for 5 faces 
result_dict = {
	'f1': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] },
	'f2': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] },
	'f3': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] },
	'f4': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] },
	'f5': {"FacedirH":[],"FacedirV":[],"Drowsy":[],"Distracted":[],"RotMatrix":[],"Tilt":[],"NoseVec":[] } }


# Dictionary for optional debug.txt output - please reconfigure to your needs 
debug2text = {"yaw":[],"pitch":[], "roll":[], "nose":[], "tilt":[], "out":[] } 


def scenery_handdetection(height, width, hand_landmarks):
	
	# hands
	str_hands=''
	is_below=False
	
	# 15 is the threshold and can be set dynamical tbd
	is_below = calculations.CalcFunctions.checkHandDistraction(height, width, hand_landmarks,15)

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
		if calculations.CalcFunctions.is_between(-90, noseDirAng[0], -40):
			facedir_horiz = 1 			# right
		elif  calculations.CalcFunctions.is_between(40, noseDirAng[0], 90):
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
			if calculations.CalcFunctions.is_between(0, abs(rotangles[1]), 10):	# Yaw range
				facedir_vert = 0		# forward
			if abs(rotangles[1]) > 10:
				facedir_vert= -1		# down
		else:
			if calculations.CalcFunctions.is_between(0, abs(rotangles[1]), 25): 	# Yaw range
				facedir_vert = 1		# up
			else:
				facedir_vert= -1		# down

	elif facedir_horiz == -1: #profileleft
		if rotangles[0]<0: 					# negativ Yaw for straight/down
				facedir_vert = -1 		#up	
		else :
			if calculations.CalcFunctions.is_between(110, abs(noseDirAng[0]), 180) :		
				facedir_vert = 1 		#up	
			elif abs(noseDirAng[0])<110:
				facedir_vert = 0
	elif facedir_horiz == 1: #profileright
		if rotangles[0]<0: 					# negativ Yaw for straight
			if calculations.CalcFunctions.is_between(10, abs(rotangles[1]), 90):
				facedir_vert = 0		# forward
			elif abs(rotangles[1]) < 10:
				facedir_vert = -1		# down
		else:
			if calculations.CalcFunctions.is_between(0, abs(rotangles[1]), 45):
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
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
    resized = cv2.resize(image,dim,interpolation=inter)

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
		
		boxes, predLabelList = calc.predFacePose(annotated_image)
		blen = len(boxes)
	
		#draw rectangle, uncomment for debugging
		#cv2.rectangle(annotated_image,(int(boxes[0][0]),int(boxes[0][1])),(int(boxes[0][2]),int(boxes[0][3])),(255,0,0),2)
		
		for face_landmarks in results.multi_face_landmarks:
			faceXY = []
			image_points = np.empty((0, 2), int)
			f_arr, ip_arr, _ = calc.getImagePoints(face_landmarks, iw, ih)
			
			faceXY.append(f_arr)		
			image_points = np.append(image_points, ip_arr, axis=0)
			
			for i in image_points:
				cv2.circle(annotated_image,(int(i[0]),int(i[1])),2,(255,0,0),-1)
			
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
			rmark, lmark, tilt, faceoval_coords, aspect_ratio_indicator = calc.faceProfileExtractor('image', face_landmarks)
			
			# calculating face oval
			fmark = calculations.CalcFunctions.faceSquareExtractor(faceoval_coords)
			cv2.circle(annotated_image,(int(fmark[0]*iw),int(fmark[1]*ih)),2,(255,100,0),-1)
			cv2.circle(annotated_image,(int(fmark[2]*iw),int(fmark[3]*ih)),2,(255,100,0),-1)
			
			# draw face oval
			ovalcoords = calculations.CalcFunctions.extractCoordsFromDict(faceoval_coords,iw,ih)
	
			# verify oval coordinates
			pts = np.array(ovalcoords,np.int32)
			cv2.polylines(annotated_image, [pts], True, (255,100,100), 2)
			
			## calculate, whether a person looks to the left(right) side or looks in straight direction
			lpoint_inside_oval = cv2.pointPolygonTest(pts, (image_points[4][0], image_points[4][1]), False)
			rpoint_inside_oval = cv2.pointPolygonTest(pts, (image_points[5][0], image_points[5][1]), False)
			
			# nose vector 2d representation
			p1, p2, rotation_vector, rotangles = calc.noseVector(faceXY, image_points, frame.shape)
			cv2.line(annotated_image, p1, p2, (238, 255, 0), 3)

			noseDirAng = calc.calculate_spatial_angles(p1, p2)

			# calculate and draw hand positions
			hand_landmarks = calc.detect_hands(hands, annotated_image)
			
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
					cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
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
				cv2.putText(annotated_image,str(face_count+1), (int(fmark[0]*iw), int(fmark[1]*ih)), cv2.FONT_HERSHEY_PLAIN, 2, (205,108,0),3)
			else:
				cv2.putText(annotated_image,str(face_count+1), (int(fmark[2]*iw), int(fmark[3]*ih)), cv2.FONT_HERSHEY_PLAIN, 2, (255,108,0),3)

			if debug_mode == 'On':
				debug2text = {'yaw': str(rotangles[0]), 'pitch': str(rotangles[1]), 'roll': str(rotangles[2]), 'nose':str(noseDirAng),'tilt':str(tilt),'out':scenery[face_count]}
				
				calculations.CalcFunctions.debugWrite(imgfilename, debug2text)	
			
			
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
				boxes, predLabelList = calc.predFacePose(frame)
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
					f_arr, ip_arr, _ = calculations.CalcFunctions.getImagePoints(face_landmarks, iw, ih)

					faceXY.append(f_arr)		
					image_points = np.append(image_points, ip_arr, axis=0)
					
					for i in image_points:
						cv2.circle(frame,(int(i[0]),int(i[1])),3,(255,0,0),-1)
						
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
					rmark, lmark, tilt, faceoval_coords, aspect_ratio_indicator = calc.faceProfileExtractor('image', face_landmarks)
			
					# calculating face oval
					fmark = calculations.CalcFunctions.faceSquareExtractor(faceoval_coords)
					cv2.circle(frame,(int(fmark[0]*iw),int(fmark[1]*ih)),2,(255,100,0),-1)
					cv2.circle(frame,(int(fmark[2]*iw),int(fmark[3]*ih)),2,(255,100,0),-1)
				
					
					# draw face oval
					ovalcoords = calculations.CalcFunctions.extractCoordsFromDict(faceoval_coords,iw,ih)
			
					# verify oval coordinates
					pts = np.array(ovalcoords,np.int32)
					cv2.polylines(frame, [pts], True, (255,100,100), 3)
				
					## calculate, whether a person looks to the left(right) side or looks in straight direction
					lpoint_inside_oval = cv2.pointPolygonTest(pts, (image_points[4][0], image_points[4][1]), False)
					rpoint_inside_oval = cv2.pointPolygonTest(pts, (image_points[5][0], image_points[5][1]), False)
					
					# nose vector 2d representation
					p1, p2, rotation_vector, rotangles = calc.noseVector(faceXY, image_points, frame.shape)
					cv2.line(frame, p1, p2, (238, 255, 0), 4)

					noseDirAng = calculations.CalcFunctions.calculate_spatial_angles(p1, p2)

					# calculate and draw hand positions
					hand_landmarks = calculations.CalcFunctions.detect_hands(hands, frame)
					
					is_below = None
					str_hands =''
					if hand_landmarks is not None:
						
						for hand_landmark in hand_landmarks:
							# Calculate the bounding box for the hands
							x_max = 0
							y_max = 0
							x_min = iw
							y_min = ih
							mp_drawing.draw_landmarks(frame, hand_landmark, calc.mp_hands.HAND_CONNECTIONS)
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
							cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
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
						cv2.putText(frame, str(face_count+1), (int(fmark[0]*iw), int(fmark[1]*ih)), cv2.FONT_HERSHEY_PLAIN, 3, (208,32,144),3)
					else:
						cv2.putText(frame, str(face_count+1), (int(fmark[2]*iw), int(fmark[3]*ih)), cv2.FONT_HERSHEY_PLAIN, 3, (208,32,144),3)
						
					face_count += 1
					faceXY = None
					image_points = None
					hand_landmarks = None
					
					# for display purpose collect results
					key = 'f'+str(face_count)
					result_dict[key] = {'FacedirH': str(facedir_horiz), 'FacedirV': str(facedir_vert), 'Drowsy': str(drowsy), 'Distracted': str(distracted), 
								'RotMatrix': str(np.round(rotation_vector, 2)), 'Tilt': str(np.round(tilt, 2)), 
								'NoseVec': str(np.round(noseDirAng, 2))}
				
								
				frame = cv2.resize(frame,(0,0), fx=0.4, fy=0.4)
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
				
					
