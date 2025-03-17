import cv2

for i in range(10):  # Check camera indexes 0 to 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
