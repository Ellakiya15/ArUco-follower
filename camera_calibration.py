import numpy as np
import cv2
import glob

# Chessboard size (number of inner corners)
chessboard_size = (8, 6)  # 8x6 means 9x7 grid points
square_size = 0.025  # Size of each square in meters (adjust if needed)

# Prepare object points (3D points in real-world coordinates)
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale to real-world size

# Arrays to store object points & image points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Open camera
cap = cv2.VideoCapture(2)  # Change index if using an external camera

num_images = 20  # Number of images to capture
captured = 0

while captured < num_images:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', frame)
        cv2.waitKey(500)  # Wait 0.5 seconds
        captured += 1
        print(f"Captured {captured}/{num_images} images")

    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 🔹 Perform camera calibration
ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 🔹 Save calibration results
np.savez("camera_calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

# 🔹 Print the results
print("\nCamera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Camera Matrix:
#  [[587.5958399    0.         316.41445283]
#  [  0.         591.19881727 210.27193803]
#  [  0.           0.           1.        ]]

# Distortion Coefficients:
#  [[-2.28714836e-01  2.38938923e+00 -1.42391818e-02 -3.03537158e-03
#   -8.58158617e+00]]

# Camera Matrix:
#  [[569.06176802   0.         329.17884264]
#  [  0.         581.52011787 262.43694995]
#  [  0.           0.           1.        ]]

# Distortion Coefficients:
#  [[-0.08179141 -0.16241733 -0.00269563  0.01108297  0.48923533]]

# web cam
# Camera Matrix:
#  [[936.99697355   0.         368.76477854]
#  [  0.         953.56781684 236.46208551]
#  [  0.           0.           1.        ]]

# Distortion Coefficients:
#  [[ 0.40616087 -2.57002766 -0.00979087  0.0137204   7.39289598]]