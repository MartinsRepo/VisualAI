import numpy as np
import math
from facenet_pytorch import MTCNN
import torch
import cv2
import mediapipe as mp

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

# 3D model points.
face3Dmodel = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left Mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
    ], dtype=np.float64)


# Define thresholds for drowsiness and distraction detection
drowsy_threshold = 150  # Adjust as needed
distracted_threshold = 15  # Adjust as needed

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160,
              margin=0,
              min_face_size=20,
              thresholds=[0.6, 0.7, 0.7], # MTCNN thresholds
              factor=0.709,
              post_process=True,
              device=device # If you don't have GPU
        )

class CalcFunctions:
    
    mp_hands = mp.solutions.hands

    @staticmethod
    def is_between(a, x, b):
        return min(a, b) < x < max(a, b)

    @staticmethod
    # converts <class 'mediapipe.framework.formats.landmark_pb2.NormalizedLandmark'> to dictionary
    def convert(NormalizedLandmark):
        res_dict = {}
        xpoints = []
        ypoints = []
        for data_point in NormalizedLandmark:
            xpoints.append(data_point.x)
            ypoints.append(data_point.y)

        res_dict["X"] = xpoints
        res_dict["Y"] = ypoints

        return res_dict

    @staticmethod
    # Landmarks: [Left Eye], [Right eye], [nose], [left mouth], [right mouth]
    def npAngle(a, b, c):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    @staticmethod
    # helper function for debugging
    def debugWrite(imgname, debug2text):
        with open('debug.txt', 'a') as f:
            f.write("\n\n")
            f.write(imgname + "\n")
            for key, value in debug2text.items():
                f.write(key + ": " + value + "\n")

    @staticmethod
    # Returns angle in radians
    def calculate_spatial_angles(p1, p2):
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]

        azimuth_radians = math.atan2(delta_x, delta_y)
        azimuth_degrees = math.degrees(azimuth_radians)

        return azimuth_degrees, azimuth_radians

    @staticmethod
    # get 2d canonical face matrix and central image points
    def eyesExtractor(right_eye_coords, left_eye_coords):
        d_right_eye_coords = CalcFunctions.convert(right_eye_coords)
        d_left_eye_coords = CalcFunctions.convert(left_eye_coords)

        r_max_x = max(d_right_eye_coords["X"])
        r_min_x = min(d_right_eye_coords["X"])
        r_max_y = max(d_right_eye_coords["Y"])
        r_min_y = min(d_right_eye_coords["Y"])

        l_max_x = max(d_left_eye_coords["X"])
        l_min_x = min(d_left_eye_coords["X"])
        l_max_y = max(d_left_eye_coords["Y"])
        l_min_y = min(d_left_eye_coords["Y"])

        r_eye_marker = [r_min_x, r_max_x, r_min_y, r_max_y]
        l_eye_marker = [l_min_x, l_max_x, l_min_y, l_max_y]

        y_deviation = r_min_y - l_min_y

        return r_eye_marker, l_eye_marker, y_deviation

    @staticmethod
    # "Eye Aspect Ratio" (EAR) introduced by Soukupová and Čech in their paper 
    # "Real-Time Eye Blink Detection Using Facial Landmarks."
    def eye_aspect_ratio(eye):
        A = np.sqrt((eye[1].x - eye[5].x) ** 2 + (eye[1].y - eye[5].y) ** 2 + (eye[1].z - eye[5].z) ** 2)
        B = np.sqrt((eye[2].x - eye[4].x) ** 2 + (eye[2].y - eye[4].y) ** 2 + (eye[2].z - eye[4].z) ** 2)
        C = np.sqrt((eye[0].x - eye[3].x) ** 2 + (eye[0].y - eye[3].y) ** 2 + (eye[0].z - eye[3].z) ** 2)

        ear = (A + B) / (2.0 * C)
        return ear

    @staticmethod
    # Face Square Extrctor function
    def faceSquareExtractor(faceoval):
        d_faceoval_coords = CalcFunctions.convert(faceoval)

        max_x = max(d_faceoval_coords["X"])
        min_x = min(d_faceoval_coords["X"])
        max_y = max(d_faceoval_coords["Y"])
        min_y = min(d_faceoval_coords["Y"])

        faceoval_marker = [min_x, min_y, max_x, max_y]

        return faceoval_marker

    @staticmethod
    # Hand detection
    def detect_hands(hands, image):
        results = hands.process(image)

        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks
        else:
            return None

    @staticmethod
    # Check if the number of landmarks below the crossline exceeds the threshold
    def checkHandDistraction(height, width, hand_landmarks, threshold):
        if hand_landmarks is None:
            return False

        bottom_right = (0, height - height // 3)
        bottom_left = (width, height - height // 3)

        crossline = np.array([bottom_left, bottom_right], dtype=int)

        landmarks_below_crossline = sum(1 for landmark in hand_landmarks[0].landmark if landmark.y * height > height // 3)

        is_below_crossline = landmarks_below_crossline >= threshold

        return is_below_crossline

    @staticmethod
    def extractCoordsFromDict(faceoval_coords, iw, ih):
        converted = CalcFunctions.convert(faceoval_coords)

        xcoords = []
        ycoords = []

        for u, v in converted.items():
            for i in range(len(v)):
                if u == 'X':
                    xcoords.append(v[i])
                if u == 'Y':
                    ycoords.append(v[i])

        result = []
        for i in range(len(xcoords)):
            x = int(xcoords[i] * iw)
            y = int(ycoords[i] * ih)
            result.append((x, y))

        return result

    @staticmethod
    def getImagePoints(landmarks, iw, ih):
        faceXY = []

        for lm in enumerate(landmarks.landmark):
            parts = lm[1]

            x = int(parts.x * iw)
            y = int(parts.y * ih)

            faceXY.append((x, y))

        image_points = np.array([
            faceXY[19],
            faceXY[152],
            faceXY[226],
            faceXY[446],
            faceXY[57],
            faceXY[287]
        ], dtype="double")

        nose_2d = faceXY[19]

        return faceXY, image_points, nose_2d

    @staticmethod
    def getAspectRatio(right_eye_landmarks, left_eye_landmarks):
        left_eye_ear = CalcFunctions.eye_aspect_ratio(left_eye_landmarks)
        right_eye_ear = CalcFunctions.eye_aspect_ratio(right_eye_landmarks)

        aspect_ratio_indicator = False
        lee = False
        ree = False

        ear[0] = int(left_eye_ear * 100)
        ear[1] = int(right_eye_ear * 100)
        for i in range(2):
            if ear[i] < drowsy_threshold:
                if i == 0:
                    lee = True
                else:
                    ree = True

        aspect_ratio_indicator = lee and ree

        return aspect_ratio_indicator

    @staticmethod
    def faceProfileExtractor(source, landmarks):
        right_eye_landmarks = [landmarks.landmark[p] for p in RIGHT_EYE]
        left_eye_landmarks = [landmarks.landmark[p] for p in LEFT_EYE]
        faceoval_coords = [landmarks.landmark[p] for p in FACE_OVAL]

        rmark, lmark, tilt = CalcFunctions.eyesExtractor(right_eye_landmarks, left_eye_landmarks)

        tilt = tilt * 100

        asr_ind = CalcFunctions.getAspectRatio(right_eye_landmarks, left_eye_landmarks)

        return rmark, lmark, tilt, faceoval_coords, asr_ind

    @staticmethod
    def predFacePose(frame):
        bbox_, prob_, landmarks_ = mtcnn.detect(frame, landmarks=True)
        angle_R_List = []
        angle_L_List = []
        predLabelList = []
        boxes = []

        def checkEye(landmarks):
            angR = CalcFunctions.npAngle(landmarks[0], landmarks[1], landmarks[2])
            angL = CalcFunctions.npAngle(landmarks[1], landmarks[0], landmarks[2])

            angle_R_List.append(angR)
            angle_L_List.append(angL)

            if ((int(angR) in range(30, 65)) and (int(angL) in range(30, 65))):
                predLabel = 'Frontal'
                predLabelList.append(predLabel)
            else:
                if angR < angL:
                    predLabel = 'RightProfile'
                else:
                    predLabel = 'LeftProfile'

            return predLabel

        try:
            for bbox, landmarks, prob in zip(bbox_, landmarks_, prob_):
                if bbox is not None:
                    boxes.append(bbox)
                    if prob > 0.8:
                        predLabel = checkEye(landmarks)
                    else:
                        predLabel = 'NotDetectable'
                else:
                    predLabel = 'NotFace'

                predLabelList.append(predLabel)
        except:
            pass

        return boxes, predLabelList

    @staticmethod
    def noseVector(faceXY, image_points, size):
        height, width = size[:2]
        p1 = p2 = None

        dist_coeffs = np.zeros((4, 1))
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        (success, rotation_vector, translation_vector) = cv2.solvePnP(face3Dmodel, image_points, camera_matrix, dist_coeffs)

        (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        rotationmatrix = cv2.Rodrigues(rotation_vector)[0]

        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotationmatrix)
        yaw = angles[0]
        pitch = angles[1]
        roll = angles[2]

        rotangles = (yaw, pitch, roll)

        return p1, p2, rotation_vector, rotangles

    @staticmethod
    def scenery_handdetection(height, width, hand_landmarks):
        str_hands = ''
        is_below = False

        is_below = CalcFunctions.checkHandDistraction(height, width, hand_landmarks, 15)

        if is_below:
            str_hands = "Hands below the face."
        else:
            str_hands = "Hands near the face."

        return str_hands, is_below

    