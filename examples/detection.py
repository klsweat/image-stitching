import cv2

# Load the images
image1 = cv2.imread('./images/image3.JPG', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('./images/image4.JPG', cv2.IMREAD_GRAYSCALE)

# FEATURE DETECTIION AND EXTRACTION-----------------------------------------------------

# SIFT  -----------------------------------------------------------------------
# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
#-------------------------------------------------------------------------------

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

# Draw keypoints on the images
image1_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
image2_keypoints = cv2.drawKeypoints(image2, keypoints2, None)

# Display the images with keypoints
cv2.imshow('Image 1 with Keypoints', image1_keypoints)
cv2.imshow('Image 2 with Keypoints', image2_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()