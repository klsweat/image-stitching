import cv2
import numpy as np

# Load the images
image1 = cv2.imread('./images/image1.png')
image2 = cv2.imread('./images/image2.png')

# Convert the images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# SIFT -----------------------------------------------------------------------
# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# ORB -----------------------------------------------------------------------
#uncomment the lines of code to run ORB
# Initialize the feature detector and extractor (e.g., ORB)
#orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
#keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
#keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
#-------------------------------------------------------------------------------

# SURF -----------------------------------------------------------------------
#uncomment the lines of code
# Initialize the feature detector and extractor (e.g., SURF)
#surf = cv2.xfeatures2d.SURF_create()

# Detect keypoints and compute descriptors for both images
#keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
#keypoints2, descriptors2 = surf.detectAndCompute(image2, None)
#-------------------------------------------------------------------------------


# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors using brute-force matching
matches = bf.match(descriptors1, descriptors2)

# Extract the matched keypoints
src_points = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Estimate the homography matrix using RANSAC
homography, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# Print the estimated homography matrix
print("Estimated Homography Matrix:")
print(homography)
