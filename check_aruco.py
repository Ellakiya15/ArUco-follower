import cv2
import numpy as np

# Load predefined dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Initialize ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        print("no marker is detected")

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        print(f"✅ Detected IDs: {ids.flatten()}")

    cv2.imshow("ArUco Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()