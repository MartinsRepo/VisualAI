import cv2
import os
import streamlit as st
import mediapipe as mp
import cv2 as cv
import numpy as np
import tempfile
import time
from PIL import Image

st.set_page_config(layout="wide")
st.session_state.setdefault("rows", {})
import analysis

DEMO_IMAGE = 'demo/demo.jpg'
DEMO_VIDEO = 'demo/demo.mp4'

left_placeholder, empty_placeholder, right_placeholder = st.columns([10, 1, 4])
video_frame_placeholder = st.empty()
video_text_placeholder = st.empty()
with left_placeholder:
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

def main():

	# About Page
	if app_mode == 'About':
		image = Image.open('about.png')
		col1, col2 = st.columns([1, 2])
		
		with col1:
			# Display the image in the first column
			st.markdown(
				"""
					<style>
						.image-container {
						    margin-top: 50px;  
						}
					</style>
					<div class="image-container">
					<img src="about.png" alt="">
					</div>
				""", unsafe_allow_html=True
)
			st.image(image, caption='')

		with col2:
			# Display the About message in the second column
			st.write("""
				## About VisualAI
				The purpose of this application is to interprete the states of faces in a scenario with persons
				to feed this to loacal LLM AI interfaced by Langchain.\n 
				**Google Mediapipe** is used to perform this. **Streamlit** is used for the media interface (GUI).\n
				- **Martin Hummel**,\n
				  Senior Software Engineer,\n
				  Germany/Bavaria/Regensburg\n
				- Email: jupp@linuxmail.org\n	
				- [Github](https://github.com/MartinsRepo/VisualAI) \n
				**Dec. 2023**
			""")

	# Image Page
	elif app_mode == 'Image':
	
		# cleanup
		st.cache_data.clear()
		st.cache_resource.clear()
		
		left_placeholder, empty_placeholder, right_placeholder = st.columns([2, 1, 4])	

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

		max_faces = st.sidebar.number_input('Maximum Number of Faces', value=3, min_value=1, max_value=5)
		st.sidebar.markdown('---')

		detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
		st.sidebar.markdown('---')

		## Output
		with left_placeholder:
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

			analysis.decode_image_mediapipe( out_image, results, face_count, left_placeholder, right_placeholder)
		
	# Video Page
	elif app_mode == 'Video':
		
		use_webcam = st.sidebar.button('Use Webcam')
		record = st.sidebar.checkbox("Record Video")

		if record:
			st.checkbox('Recording', True)

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

		max_faces = st.sidebar.number_input('Maximum Number of Faces', value=2, min_value=1, max_value=5)
		st.sidebar.markdown('---')
		detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.0,max_value=1.0,value=0.5)
		tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.0,max_value=1.0,value=0.5)
		st.sidebar.markdown('---')

		## Get Video	
		video_file_buffer = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'])
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
		out = cv.VideoWriter('output.mp4', codec, fps_input, (width,height))

		st.sidebar.text('Input Video')
		st.sidebar.video(temp_file.name)

		fps = 0
		i = 0
		
		analysis.decode_video_mediapipe(video, max_faces, detection_confidence, tracking_confidence)

		try:
			os.remove('output.mp4')
		except OSError as e:
			# If it fails, inform the user.
			print("Error: %s - %s." % (e.filename, e.strerror))

                
if __name__ == "__main__":
	main()
	
