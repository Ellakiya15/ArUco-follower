# import numpy as np
# import cv2
# import cv2.aruco

# ARUCO_DICT= {
#     "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
#     "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
#     "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
#     "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
#     "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
#     "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
#     "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#     "DICT_APRILTAG_16h5" : cv2.aruco.DICT_APRILTAG_16h5,
# }

# aruco_type = "DICT_4X4_100"
# id = 1

# arucoDict = cv2.aruco.Dictionary(ARUCO_DICT[aruco_type])

# print("Aruco type '{}' with ID '{}'".format(aruco_type,id))
# tag_size = 500

# tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
# cv2.aruco.drawMarker(arucoDict,id,tag_size,tag, 1)

# tag_name = "arucoMarkers/" + aruco_type + "_" + str(id) + ".png"
# cv2.imwrite(tag_name,tag)
# cv2.imshow("Aruco Tag", tag)

# cv2.waitKey(0)

# cv2.destroyAllWindows()
import numpy as np
import cv2
import cv2.aruco

# print("OpenCV Version:", cv2.__version__)

# Define available ArUco dictionaries
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# # Check if AprilTag support exists in OpenCV 4.7+
# if hasattr(cv2.aruco, "DICT_APRILTAG_16h5"):
#     ARUCO_DICT["DICT_APRILTAG_16h5"] = cv2.aruco.DICT_APRILTAG_16h5

aruco_type = "DICT_6X6_50"  # Change if unsupported
id = 1

# Ensure the dictionary exists
if aruco_type not in ARUCO_DICT:
    raise ValueError(f"ArUco dictionary '{aruco_type}' is not available in your OpenCV version.")

# Correct method to get predefined dictionary
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

print(f"Generating ArUco tag: {aruco_type} with ID {id}")

# Generate and save the marker
tag_size = 500
tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")

# Use the new method for drawing markers in OpenCV 4.10.0+
if hasattr(cv2.aruco, "generateImageMarker"):
    tag = cv2.aruco.generateImageMarker(arucoDict, id, tag_size)
else:
    cv2.aruco.drawMarker(arucoDict, id, tag_size, tag, 1)  # Fallback for older versions

tag_name = f"arucoMarkers/{aruco_type}_{id}.png"
cv2.imwrite(tag_name, tag)
print(f"Marker saved as {tag_name}")

cv2.imshow("Aruco Tag", tag)
cv2.waitKey(0)
cv2.destroyAllWindows()


