import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1 = cv2.imread('./images/image3.JPG')
image2 = cv2.imread('./images/image4.JPG')

# Convert images to grayscale
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
# Initialize the feature detector and extractor (e.g., ORB)
#surf = cv2.SURF_create()

# Detect keypoints and compute descriptors for both images
#keypoints1, descriptors1 = surf.detectAndCompute(image1, None)
#keypoints2, descriptors2 = surf.detectAndCompute(image2, None)
#-------------------------------------------------------------------------------

# Initialize the feature matcher using brute-force matching
bf = cv2.BFMatcher()

# Match the descriptors using brute-force matching
#matches = bf.match(descriptors1, descriptors2)

#perform ratio test to get best matches and remove invalid
matches = bf.knnMatch(descriptors1,descriptors2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m);

# Select the top N matches
num_matches =  len(good)
matches = sorted(good, key = lambda x:x.distance)

# Extract matching keypoints
src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
# Draw first 10 matches.
img3 = cv2.drawMatches(image1,keypoints1,image2,keypoints2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
 
#plt.imshow(img3),plt.show()

# Estimate the homography matrix
homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# Warp the first image using the homography
result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

# Blending the warped image with the second image using alpha blending
alpha = 0.5  # blending factor
blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)

# Display the blended image
cv2.imshow('Blended Image', blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
