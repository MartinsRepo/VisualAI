import cv2
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image

import analysis

DEMO_IMAGE = 'demo/demo.jpg'
DEMO_VIDEO = 'demo/demo.mp4'

# Basic App Scaffolding
st.title('Face Detection with Mediapipe')

## Add Sidebar and Main Window style
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    </style>
    """,
    unsafe_allow_html=True,
)

## Create Sidebar
st.sidebar.title('Selection')
st.sidebar.subheader('Parameter')

## Define available pages in selection box
app_mode = st.sidebar.selectbox(
    'App Mode',
    ['Image','Video','About']
)

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
    # calculate the ratio of the width and construct the
    # dimensions
    else:
        r = width/float(w)
        dim = width, int(h*r)

    # Resize image
    resized = cv.resize(image,dim,interpolation=inter)

    return resized


def disp_res(source, kpil1_text, kpil2_text, face_count, fps):
	if source == 'image':
		kpil1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
	elif source == 'video':
		kpil1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
		kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)



def main():
	# About Page
	if app_mode == 'About':
		st.markdown('''
			## Face Mesh \n
			In this application we are using **MediaPipe** for creating a Face Mesh. **StreamLit** is to create 
			the Web Graphical User Interface (GUI) \n
			
			- [Github](https://github.com/mpolinowski/streamLit-cv-mediapipe) \n
		''')

		## Add Sidebar and Window style
		st.markdown(
			"""
				<style>
				[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
				    width: 350px
				}
				[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
				    width: 350px
				    margin-left: -350px
				}
				</style>
			""",
			unsafe_allow_html=True,
		)

	# Image Page

	elif app_mode == 'Image':
		drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

		st.sidebar.markdown('---')

		## Add Sidebar and Window style
		st.markdown(
			"""
				<style>
				[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
				    width: 350px
				}
				[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
				    width: 350px
				    margin-left: -350px
				}
				</style>
			""",
			unsafe_allow_html=True,
		)

		st.markdown("**Detected Faces**")
		kpil1_text = st.markdown('0')

		max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1)
		st.sidebar.markdown('---')

		detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
		st.sidebar.markdown('---')

		## Output
		st.markdown('## Output Image')
		img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
		
		if img_file_buffer is not None:
			image = np.array(Image.open(img_file_buffer))

		else:
			demo_image = DEMO_IMAGE
			image = np.array(Image.open(demo_image))

		st.sidebar.text('Original Image')
		st.sidebar.image(image)

		face_count=0

		## Dashboard
		with mp.solutions.face_mesh.FaceMesh(
			static_image_mode=True, #Set of unrelated images
			max_num_faces=max_faces,
			min_detection_confidence=detection_confidence,
			refine_landmarks=False,
			min_tracking_confidence=0.01
		) as face_mesh:

			results = face_mesh.process(image)
			out_image=image.copy()

			face_count = analysis.decode_mediapipe('image', out_image, results, face_count, None)
			
			#kpil1_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
			disp_res('image', kpil1_text, None, face_count, None)

	# Video Page

	elif app_mode == 'Video':


		use_webcam = st.sidebar.button('Use Webcam')
		record = st.sidebar.checkbox("Record Video")

		if record:
			st.checkbox('Recording', True)

		drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

		st.sidebar.markdown('---')

		## Add Sidebar and Window style
		st.markdown(
			"""
				<style>
				[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
				    width: 350px
				}
				[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
				    width: 350px
				    margin-left: -350px
				}
				</style>
			""",
			unsafe_allow_html=True,
		)

		max_faces = st.sidebar.number_input('Maximum Number of Faces', value=5, min_value=1)
		st.sidebar.markdown('---')
		detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
		tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0,max_value=1.0,value=0.5)
		st.sidebar.markdown('---')

		## Get Video
		stframe = st.empty()
		video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
		temp_file = tempfile.NamedTemporaryFile(delete=False)

		if not video_file_buffer:
			if use_webcam:
				video = cv.VideoCapture(0)
			else:
				video = cv.VideoCapture(DEMO_VIDEO)
				temp_file.name = DEMO_VIDEO

		else:
			temp_file.write(video_file_buffer.read())
			video = cv.VideoCapture(temp_file.name)

		width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
		height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
		fps_input = int(video.get(cv.CAP_PROP_FPS))

		## Recording
		codec = cv.VideoWriter_fourcc(*"mp4v")
		out = cv.VideoWriter('output1.mp4', codec, fps_input, (width,height))

		st.sidebar.text('Input Video')
		st.sidebar.video(temp_file.name)

		fps = 0
		i = 0

		drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=1)

		kpil1, kpil2, kpil3 = st.columns(3)

		with kpil1:
			st.markdown('**Frame Rate**')
			kpil1_text = st.markdown('0')

		with kpil2:
			st.markdown('**Detected Faces**')
			kpil2_text = st.markdown('0')

		st.markdown('<hr/>', unsafe_allow_html=True)


		## Face Mesh
		with mp.solutions.face_mesh.FaceMesh(
			max_num_faces=max_faces,
			min_detection_confidence=detection_confidence,
			min_tracking_confidence=tracking_confidence

		) as face_mesh:

			prevTime = 0

			while video.isOpened():
				i +=1
				ret, frame = video.read()
				if not ret:
					continue

				results = face_mesh.process(frame)
				frame.flags.writeable = True

				face_count = 0

				face_count = analysis.decode_mediapipe('video', frame, results, face_count, drawing_spec)


				# FPS Counter
				currTime = time.time()
				fps = 1/(currTime - prevTime)
				prevTime = currTime

				if record:
					out.write(frame)

		        	# Dashboard
				#kpil1_text.write(f"<h1 style='text-align: center; color:red;'>{int(fps)}</h1>", unsafe_allow_html=True)
				#kpil2_text.write(f"<h1 style='text-align: center; color:red;'>{face_count}</h1>", unsafe_allow_html=True)
				disp_res('video', kpil1_text, kpil2_text, face_count, fps)


				frame = cv.resize(frame,(0,0), fx=0.8, fy=0.8)
				frame = image_resize(image=frame, width=640)
				stframe.image(frame,channels='BGR', use_column_width=True)

                
if __name__ == "__main__":
	main()

