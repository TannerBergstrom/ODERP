import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Author: Tanner Bergstrom. Date 2025/08/04
# The goal of this project is to find the distance of an object from a camera in an image.
# A camera projects a snippet of a 3D environment onto a 2D image, eliminating the z-axis, or depth, in the process.
# The major hurdle in reaching the goal of this project is approximating depth in an image after it has been eliminated.
# For approximation, we need context. This context comes from another image. This other image, or reference image,
# captures a key reference object. The importance of the reference image is that the real-world distance between the
# camera and the reference object is known. With this reference object, we can find the depth of this object in other
# images.
# This approach uses SIFT descriptor detection, FLANN-based descriptor matching, and image homography to map a
# reference object to its location within another image and find the distance of the object
# in that image from the camera through an angular size comparison and focal length conversion. This is realized
# through this toy program, though the idea is for use as a mobile app. This approach is called ODERP.
#
# OpenCV reference: https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html

referenceDistance = input("enter the distance between the reference object and camera in cm: ")

# File Picking
print("Choose the reference image")
Tk().withdraw()
filename = askopenfilename()
imgReference = cv.imread(filename, cv.IMREAD_GRAYSCALE)

print("Choose the destination image")
Tk().withdraw()
filename = askopenfilename()
destImg = cv.imread(filename, cv.IMREAD_GRAYSCALE)

zoomScale = input("enter the camera zoom: ")

# Mouse Event for cropping
def mouse_Click(event, x, y, flags, param):
    global xMouse
    global yMouse
    global click
    if event == cv.EVENT_LBUTTONDOWN:
        xMouse = x
        yMouse = y
        click += 1

xMouse = -1
yMouse = -1
click = 0

# Resizing the image window for cropping
userWindowHeight = 500
queryH, queryW = imgReference.shape
scale = userWindowHeight / queryH
userCropImg = cv.resize(imgReference, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

# Window Text
text = "Click the top-left, then the bottom-right of the reference object."
cv.putText(userCropImg, text, org = (25, 25), fontFace = cv.FONT_HERSHEY_SIMPLEX,
           fontScale = .25, color = (0, 0, 0), thickness = 1, lineType = cv.LINE_AA)

# Show image
cv.imshow("image", userCropImg)
cv.namedWindow('image')
cv.setMouseCallback('image', mouse_Click)

# Mouse coordinates for cropping
coord1 = [xMouse, yMouse]
coord2 = [xMouse, yMouse]

while (1):
    if click == 1:
        coord1 = [xMouse, yMouse]
    if click == 2:
        coord2 = [xMouse, yMouse]
        break
    if cv.waitKey(33) == ord('d'):
        break


cv.destroyAllWindows()

# scale coordinates to original image size
coord1[1] = int(coord1[1] /scale)
coord1[0] = int(coord1[0] /scale)
coord2[1] = int(coord2[1] /scale)
coord2[0] = int(coord2[0] /scale)

referenceObjImg = imgReference[coord1[1]:coord2[1], coord1[0]:coord2[0]]

MIN_MATCH_COUNT = 10

# Initiate SIFT detector
sift = cv.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(referenceObjImg, None)
kp2, des2 = sift.detectAndCompute(destImg, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# Find two nearest neighbors for each descriptor
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Obtain transformation matrix for perspective projection of the reference object onto image
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    referenceObjHeight, referenceObjWidth = referenceObjImg.shape
    pts = np.float32([[0, 0], [0, referenceObjHeight - 1], [referenceObjWidth - 1, referenceObjHeight - 1], [referenceObjWidth - 1, 0]]).reshape(-1, 1, 2)
    destCoords = cv.perspectiveTransform(pts, M)

    destImg = cv.polylines(destImg, [np.int32(destCoords)], True, 255, 3, cv.LINE_AA)

    # Find the average height in pixels of the object in the destination image
    heightDst = ((destCoords[1][0][1] - destCoords[0][0][1]) + (destCoords[2][0][1] - destCoords[3][0][1])) / 2

    destinationDistance = (float)(referenceObjHeight / heightDst) * (float)(referenceDistance) * (float)(zoomScale)

    print("The object is ", destinationDistance, "cm away")

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None


draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask,  # draw only inliers
                   flags=2)

img3 = cv.drawMatches(referenceObjImg, kp1, destImg, kp2, good, None, **draw_params)

plt.imshow(img3, 'gray'), plt.show()