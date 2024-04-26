import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1 = cv2.imread('./images/image3.JPG', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./images/image4.JPG', cv2.IMREAD_GRAYSCALE)

# FEATURE DETECTIION AND EXTRACTION-----------------------------------------------------

# SIFT  -----------------------------------------------------------------------
# Initialize the feature detector and extractor (e.g., SIFT)
#sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
#keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
#keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
#index_params = dict(algorithm=0, trees=5)

#-------------------------------------------------------------------------------

# ORB -----------------------------------------------------------------------
#uncomment the lines of code to run ORB
# Initialize the feature detector and extractor (e.g., ORB)
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
table_number = 6, # 12
key_size = 12, # 20
multi_probe_level = 1) #2
#-------------------------------------------------------------------------------

# SURF -----------------------------------------------------------------------
#uncomment the lines of code
# Initialize the feature detector and extractor (e.g., ORB)
#surf = cv2.SURF_create()

# Detect keypoints and compute descriptors for both images
#keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
#keypoints2, descriptors2 = surf.detectAndCompute(image2, None)
#-------------------------------------------------------------------------------

# Draw keypoints on the images
image1_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
image2_keypoints = cv2.drawKeypoints(image2, keypoints2, None)

# Display the images with keypoints
#cv2.imshow('Image 1 with Keypoints', image1_keypoints)
#cv2.imshow('Image 2 with Keypoints', image2_keypoints)

# FEATURE MATCHING ----------------------------------------------------------------
# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match the descriptors using brute-force matching
matches_bf = bf.match(descriptors1, descriptors2)

# Sort the matches by distance (lower is better)
matches_bf = sorted(matches_bf, key=lambda x: x.distance)

# Draw the top N matches
num_matches = 50
image_matches_bf = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_bf[:num_matches], None)

print('brute', len(matches_bf))


# Initialize the feature matcher using FLANN matching
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match the descriptors using FLANN matching
matches_flann = flann.match(descriptors1, descriptors2)

# Sort the matches by distance (lower is better)
matches_flann = sorted(matches_flann, key=lambda x: x.distance)

# Draw the top N matches
image_matches_flann = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches_flann[:num_matches], None)
print('FLANN', len(matches_flann))

# Display the images with matches
#cv2.imshow('Brute-Force Matching', image_matches_bf)
#cv2.imshow('FLANN Matching', image_matches_flann)
plt.title("BRUTE NO RATIO")
plt.imshow(image_matches_bf,),plt.show()
plt.title("FLANN NO RATIO")
plt.imshow(image_matches_flann,),plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()


