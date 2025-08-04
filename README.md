Object Distance Estimation in an Image from a Reference Photo
By Tanner Bergstrom
August 4th, 2025

The goal of this project is to find the distance of an object from a camera in an image. A camera projects a snippet of a 3D environment onto a 2D image, eliminating the z-axis, or depth, in the process. 
The major hurdle in reaching the goal of this project is approximating depth in an image after it has been eliminated.
For approximation, we need context. This context comes from another image. This other image, or reference image, captures a key reference object. The importance of the reference image is that the real-world 
distance between the camera and the reference object is known. With this reference object, we can find the depth of this object in other images. 
The following analysis presents an approach using SIFT descriptor detection, FLANN-based descriptor matching, and image homography to map a reference object to its location within another image and find the 
distance of the object in that image from the camera through an angular size comparison and focal length conversion. This is realized through a toy program, though the idea is for use as a mobile app. This 
approach will be referred to as object distance estimation from a reference photo, or ODERP.
