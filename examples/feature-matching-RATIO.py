import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1 = cv2.imread('./images/image3.JPG')
image2 = cv2.imread('./images/image4.JPG')

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.IMREAD_GRAYSCALE)
gray2 = cv2.cvtColor(image2, cv2.IMREAD_GRAYSCALE)

# SIFT -----------------------------------------------------------------------
# Initialize the feature detector and extractor (e.g., SIFT)
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

# ORB -----------------------------------------------------------------------
#uncomment the lines of code to run ORB
# Initialize the feature detector and extractor (e.g., ORB)
#orb = cv2.ORB_create()

# Detect keypoints and compute descriptors for both images
#keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
#keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

#FLANN_INDEX_LSH = 6
#index_params= dict(algorithm = FLANN_INDEX_LSH,
# table_number = 6, # 12
# key_size = 12, # 20
# multi_probe_level = 1) #2
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
brute = cv2.drawMatches(image1,keypoints1,image2,keypoints2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
print('brute', len(matches))

plt.title("BRUTE")
plt.imshow(brute),plt.show()

# Initialize the feature matcher using FLANN matching
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
matches_flann = flann.knnMatch(descriptors1,descriptors2,k=2)
 
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches_flann))]
 
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches_flann):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
 
draw_params = dict(matchColor = (0,255,0),
 singlePointColor = (255,0,0),
 matchesMask = matchesMask,
 flags = cv2.DrawMatchesFlags_DEFAULT)
 
img3 = cv2.drawMatchesKnn(image1,keypoints1,image2,keypoints2,matches_flann,None,**draw_params)
print('flann', len(matchesMask))
plt.title("FLANN")
plt.imshow(img3,),plt.show()