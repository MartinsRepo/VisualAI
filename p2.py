#ip install Selenium
#pip install flask
#pip install opencv-python

from flask import Flask, render_template, request, Response
import socket
from threading import Timer
from selenium import webdriver
import cv2

app2 = Flask(__name__)


@app2.route('/')
def index():
	return render_template('indexTrafficmon.html')

@app2.route('/video_feed')
def video_feed():
	camera = cv2.VideoCapture(0)  # Open the camera
	if camera.isOpened():
		ret, frame = camera.read()  # Read a frame from the camera
		
		# Set the response content type to be MJPEG
		response = Response(generate_frames(camera),
				mimetype='multipart/x-mixed-replace; boundary=frame')
	else:
		response = Response(status=204)
	return response
	
@app2.route('/radio-status', methods=['POST'])
def radio_status():
	status = request.form.get('status')
	# Send status to p2.py
	send_status_to_p1(status)
	return 'OK'


def send_status_to_p1(status):
	# Establish a socket connection to p2.py
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.connect(('localhost', 5002))
	# Send the status to p2.py
	sock.sendall(status.encode())
	sock.close()


def receive_status_from_p1():
	# Set up a socket to receive status from p1.py
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.bind(('localhost', 5001))
	sock.listen(1)

	while True:
		conn, addr = sock.accept()
		data = conn.recv(1024).decode()
		# Process the received status from p1.py
		process_status(data)
		conn.close()
	
	
def process_status(status):
	# Do something with the received status
	print(f"Received status: {status}")


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
    
    
def open_browser(url):
	driver = webdriver.Firefox()
	driver.get(url)
	return driver


# Define functions for each process
def process2():
	global drvmontab
	url = "http://localhost:5002"
	trafficmontab = Timer(1, open_browser, args=(url,))
	trafficmontab.start()

	# Start Flask server on port 5001
	app2.run(port=5002)

if __name__ == '__main__':
	process2()
	trafficmontab.quit()
