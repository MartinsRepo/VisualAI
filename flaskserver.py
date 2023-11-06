#ip install Selenium
#pip install flask[async]
#pip install opencv-python
#pip install Pillow
#pip install mediapipe 
#pip install asyncio

#Convert videos in readable format
#ffmpeg -i distracted2.mp4 -c:v libx264 -c:a aac -strict -2 adistracted2.mp4

from flask import Flask, render_template, request, Response, jsonify, send_file, make_response, url_for
import socket
from threading import Timer
from selenium import webdriver
import os
import asyncio
import cv2
from PIL import Image
import io
from queue import Empty 
from queue_manager import queueConfig
from queue_manager import queueFName
from queue_manager import queueModifiedImage
from queue_manager import queueScenery

app1 = Flask(__name__)

UPLOAD_FOLDER_IMG =  os.path.join('images/drvmonpics')
app1.config['UPLOAD_FOLDER_IMG'] = UPLOAD_FOLDER_IMG
UPLOAD_FOLDER_VID =  os.path.join('videos/person')
app1.config['UPLOAD_FOLDER_VID'] = UPLOAD_FOLDER_VID

#app1._static_folder = UPLOAD_FOLDER_IMG

initial_filenames = {
    'image': ['initial.jpeg'],
    'video': ['awake.mp4'],
}

@app1.route('/')
def index():
	return render_template('indexDrvmon.html')
	
@app1.route("/favicon.ico")
def favicon():
	return url_for('static', filename='data:,')


@app1.route('/image_feed')
def image_feed():
	image_path = os.path.join(app1.config['UPLOAD_FOLDER_IMG'], 'initial.jpeg')
	
	fname = {'filename': 'initial.jpeg_1'}
        
	setQueuingFName(fname)

	# Return the HTML code to display the image
	return send_file(image_path, mimetype='image/jpeg')


@app1.route('/get-filenames', methods=['POST'])
def get_initial_filenames():
	data = request.get_json()
	media_type = data.get('media_type')

	if media_type in initial_filenames:
		filenames = initial_filenames[media_type]
		#print('Received filename:', filenames)
        
		fname = {'filename': filenames}
        
		setQueuingFName(fname)
        
		return  jsonify(fname)
	else:
		return jsonify({'error': 'Invalid media type'})

    
@app1.route('/get-modified-image', methods=['GET'])
def get_modified_image():
	response = None
	while True:
		if checkqueueModImage():
		
			modified_image = getQueuingModImage()
			
			# Convert the NumPy ndarray to a PIL image
			pil_image = Image.fromarray(modified_image)
			#pil_image.show()

			# Create an in-memory binary stream to store the image data
			img_io = io.BytesIO()
			pil_image.save(img_io, 'JPEG') 
			img_io.seek(0)
			
			response = make_response(send_file(img_io, mimetype='image/jpg'))
			response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'

			break
	return response


@app1.route('/get_information', methods=['GET'])
def get_information():
	if checkqueueScenery():
		information = getQueuingScenery()
		
		text = {'textoutput': information}
        
		return  jsonify(text)
	else:
		return "No information available"


@app1.route("/video_feed")
def video_feed():
    video_path = os.path.join(app1.config['UPLOAD_FOLDER_VID'], 'awake.mp4')

    # Return the HTML code to display the image
    return send_file(video_path, mimetype='video/mp4')

   
@app1.route('/camera_feed')
def camera_feed():
	camera = cv2.VideoCapture(0)  # Open the camera
	if camera.isOpened():
		ret, frame = camera.read()  # Read a frame from the camera
		
		# Set the response content type to be MJPEG
		response = Response(generate_frames(camera),
				mimetype='multipart/x-mixed-replace; boundary=frame')
	else:
		response = Response(status=204)
	return response
	
	
@app1.route('/radio-status', methods=['POST'])
def handle_radio_status():
	drvmon = request.form.get('drvmon')  # Get the selected radio button's value
    
	# You can now use 'drvmon' in your Python code
	print("Selected value:", drvmon)
    
	# Add your logic here
    
	# Respond with a simple success message
	return jsonify({"message": "Status updated successfully"})
 
 
@app1.route('/capture-inputs', methods=['POST'])
def capture_inputs():
	drowsiness = request.form.get('drowsiness')
	awake = request.form.get('awake')
	distraction = request.form.get('distraction')
	smile = request.form.get('smile')
	
	data =	{
	  "drowsiness": drowsiness,
	  "awake": awake,
	  "distraction": distraction,
	  "smile": smile
	}
	
	setQueuingConfig(data)
	
	# Respond with a success message
	return jsonify({"message": "Inputs captured successfully"})   
	
	
@app1.route('/upload-filename', methods=['POST'])
async def upload_filename():

	data = request.get_json()
	filename = data.get('filename')
	
	data =	{
	  "filename": filename
	}
	
	# Simulate a delay or time-consuming operation 
	await asyncio.sleep(1)
	
	setQueuingFName(data)
	
	#print('Received filename:', filename)

	# Respond with a success message
	return jsonify({"message": "Filename received successfully"})

def generate_frames(camera):
	while True:
		if camera==None:
			break
		ret, frame = camera.read()  # Read a frame from the camera
		if not ret:
			break
		else:
			# Convert the frame to JPEG format
			ret, buffer = cv2.imencode('.jpg', frame)
			frame = buffer.tobytes()

		# Yield the frame in a multipart response
		yield (b'--frame\r\n'
			b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
	camera.release()



# setting uploaded config data from Flask 
def setQueuingConfig(data):
	queueConfig.put(data)
	

# setting uploaded filename from Flask 	
def setQueuingFName(data):
	queueFName.put(data)
	
	
# setting uploaded filename from Flask 	
def setQueuingModifiedImage(data):
	queueModifiedImage.put(data)


# getting modifyied image from p1 	
def getQueuingModImage():
	data = None
	try:
		data = queueModifiedImage.get_nowait()
		#clear queue
		while not queueModifiedImage.empty():
			queueModifiedImage.get()
			
	except queueModifiedImage.empty():
		pass
		
	return data
	
	
def checkqueueModImage():
	ret = False
	if not queueModifiedImage.empty():
		ret = True
	return ret
	
	
# getting scenery from p1 	
def getQueuingScenery():
	data = None
	try:
		data = queueScenery.get()
		#print('#',data)
		#clear queue
		while not queueScenery.empty():
			queueScenery.get()
			
	except queueScenery.empty():
		pass
		
	return data


def checkqueueScenery():
	ret = False
	if not queueScenery.empty():
		ret = True
	return ret


def open_browser(url):
	driver = webdriver.Firefox()
	driver.get(url)
	return driver


# Start Flsk Server und media/cam handling
def flaskprocess():
	global drvmontab
	url = "http://localhost:5001"
	drvmontab = Timer(1, open_browser, args=(url,))
	drvmontab.start()
	
	# Start Flask server on port 5001
	app1.run(port=5001)


if __name__ == '__main__':
	flaskprocess()
#	#flaskserver.quit()
