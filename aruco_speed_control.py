import cv2
import numpy as np

# Load camera parameters
camera_matrix = np.array([[569.06176802, 0, 329.17884264], 
                          [0, 581.52011787, 262.43694995], 
                          [0, 0, 1]])

dist_coeffs = np.array([[-0.08179141, -0.16241733, -0.00269563, 0.01108297, 0.48923533]])

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

# Function to calculate motor speed based on distance
def calculate_motor_speed(distance):
    if distance > 1.5:  # Marker is far
        return 100  # High RPM
    elif 0.5 < distance <= 1.5:  # Marker is at medium distance
        return 60  # Medium RPM
    else:  # Marker is too close
        return 0  # Stop

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            # SolvePnP for pose estimation
            _, rvec, tvec = cv2.solvePnP(marker_corners_3D, corners[i][0], camera_matrix, dist_coeffs)

            # Extract the Z-value (distance from the camera)
            distance = tvec[2][0]
            print(f"Distance: {distance:.2f} meters")

            # Get motor speed based on distance
            motor_speed = calculate_motor_speed(distance)
            print(f"Motor Speed: {motor_speed} RPM")

            # Send speed command to motor (replace with actual motor control function)
            # Example: motor.set_speed(motor_speed)

            # Draw marker and axis
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

    cv2.imshow("ArUco Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
