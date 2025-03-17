import cv2
import numpy as np

# Load camera parameters
camera_matrix = np.array([[569.06176802, 0, 329.17884264],
                          [0, 581.52011787, 262.43694995],
                          [0, 0, 1]], dtype=np.float32)

dist_coeffs = np.array([[-0.08179141, -0.16241733, -0.00269563, 0.01108297, 0.48923533]], dtype=np.float32)

# Define ArUco dictionary and detector parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Initialize ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Marker size (in meters)
marker_size = 0.05  

# Define 3D coordinates of the marker's four corners
marker_corners_3D = np.array([
    [-marker_size / 2, marker_size / 2, 0],
    [marker_size / 2, marker_size / 2, 0],
    [marker_size / 2, -marker_size / 2, 0],
    [-marker_size / 2, -marker_size / 2, 0]
], dtype=np.float32)

# Open webcam
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        print(ids)
        for i in range(len(ids)):
            # SolvePnP for pose estimation
            _, rvec, tvec = cv2.solvePnP(marker_corners_3D, corners[i][0], camera_matrix, dist_coeffs)

            # Draw marker and axis
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    cv2.imshow("ArUco Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# import numpy as np
# import cv2
# import sys
# import time

# ARUCO_DICT = {
#     "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
#     "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
#     "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
#     "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
#     "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
#     "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
#     "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
# }

# def aruco_display(corners, ids, rejected, image):

#     if len(corners) > 0:

#         ids = ids.flatten()

#         for (markerCorner, markerID) in zip(corners, ids):
#             corners = markerCorner.reshape((4,2))
#             (topLeft, topRight, bottomRight, bottomLeft) = corners

#             topRight = (int(topRight[0]), int(topRight[1]))
#             bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
#             bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
#             topLeft = (int(topLeft[0]), int(topLeft[1]))

#             cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
#             cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
#             cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
#             cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

#             cX = int((topLeft[0] + bottomRight[0]) / 2.0)
#             cY = int((topLeft[1] + bottomRight[1]) / 2.0)
#             cv2.circle(image, (cX, cY), 4, (0,0,255), -1)

#             cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_COMPLEX :
#                         0.5, (0, 255, 0), 2)
#             print("[Inference] ArUco marker ID:  {}".format(markerID))
#     return image

# def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.aruco_dist = cv2.aruco.Dictionary(aruco_dict_type) #Dictionary_get
#     parameters = cv2.aruco.DetectorParameters() #DetectorParameters_create()

#     corners, ids, rejected_img_points = cv2.aruco.drawDetectedMarkers(gray, cv2.aruco_dist, parameters=parameters,
#         cameraMatrix=matrix_coefficients,
#         distCoeff=distortion_coefficients)
    
#     if len(corners) > 0:
#         for i in range (0, len(ids)):

#             rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
#                                                                            distortion_coefficients)
#             cv2.aruco.drawDetectedMarkers(frame, corners)
#             cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

#     return frame

# aruco_type = "DICT_4X4_100"

# arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[aruco_type])

# arucoParams = cv2.aruco.DetectorParameters_create()

# intrinsic_camera = np.array(((587.5958399, 0, 316.41445283),(0, 591.19881727, 210.27193803),(0, 0, 1)))


# distortin = np.array((-2.28714836e-01 ,2.38938923e+00, -1.42391818e-02 ,-3.03537158e-03, -8.58158617e+00))

# cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# while cap.isOpened():
#     ret, img = cap.read()
#     output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortin)

#     cv2.imshow('Estimated Pose', output)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# import numpy as np
# import cv2
# import sys
# import time

# ARUCO_DICT = {
#     "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
#     "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
#     "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
#     "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
#     "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
#     "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
#     "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
# }

# def aruco_display(corners, ids, rejected, image):
#     if len(corners) > 0:
#         ids = ids.flatten()
#         for (markerCorner, markerID) in zip(corners, ids):
#             corners = markerCorner.reshape((4,2))
#             (topLeft, topRight, bottomRight, bottomLeft) = corners

#             topRight = (int(topRight[0]), int(topRight[1]))
#             bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
#             bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
#             topLeft = (int(topLeft[0]), int(topLeft[1]))

#             cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
#             cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
#             cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
#             cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

#             cX = int((topLeft[0] + bottomRight[0]) / 2.0)
#             cY = int((topLeft[1] + bottomRight[1]) / 2.0)
#             cv2.circle(image, (cX, cY), 4, (0,0,255), -1)

#             cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 
#                         0.5, (0, 255, 0), 2)
#             print("[Inference] ArUco marker ID:  {}".format(markerID))
#     return image

# def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
#     parameters = cv2.aruco.DetectorParameters()

#     corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

#     if ids is not None:
#         for i in range(len(ids)):
#             rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
#             cv2.aruco.drawDetectedMarkers(frame, corners)
#             cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

#     return frame

# aruco_type = "DICT_4X4_100"
# arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
# arucoParams = cv2.aruco.DetectorParameters()

# intrinsic_camera = np.array([[587.5958399, 0, 316.41445283],
#                              [0, 591.19881727, 210.27193803],
#                              [0, 0, 1]])

# distortion = np.array([-0.228714836, 2.38938923, -0.0142391818, -0.00303537158, -8.58158617])

# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         break

#     output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)
#     cv2.imshow('Estimated Pose', output)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

