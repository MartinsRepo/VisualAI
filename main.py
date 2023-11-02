# Import modules
import multiprocessing
from p1 import mediapipeprocess
from flaskserver import flaskprocess
import queue_manager

if __name__ == "__main__":

	try:
		multiprocessing.set_start_method("forkserver")  # Use the "forkserver" method if not already set
	except RuntimeError:
		pass  # Ignore if the context has already been set

	# Create and start processes
	drvmonanalysis = multiprocessing.Process(target=mediapipeprocess)
	drvmonanalysis.start()
	drvmonserver = multiprocessing.Process(target=flaskprocess)
	drvmonserver.start()
	
	# Keep the main process alive until the user presses `Ctrl`+`C`
	try:
		while True:
			pass
	except KeyboardInterrupt:
		print("Terminating processes...")
		drvmonserver.terminate()
		drvmonanalysis.terminate()
		drvmonserver.join()
		drvmonanalysis.join()


