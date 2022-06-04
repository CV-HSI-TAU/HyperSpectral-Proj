import os
import spectral.io.envi as envi
import spectral as sp
import cv2
import torch
import numpy as np
from torchvision import transforms

# def Registration_And_Homography (src_path , dst_path, name):
def Find_Homography (src_path , dst_path):
    # Open the image files.
    img1_color = cv2.imread(dst_path) # Image to be aligned.
    img2_color = cv2.imread(src_path) # Reference image.

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    # (which is not required in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches = list(matches)
    matches.sort(key = lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)
    return homography, height, width


def Registration_H(dst_path, homography, height, width, path_to_save):

    img1_color = cv2.imread(dst_path) # Image to be aligned.

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
            homography, (width, height))

    # Save the output.
    cropped_image = transformed_img[500:1100,1150:1650]
    cv2.imwrite(path_to_save , cropped_image)

# Transforming the monos

name = 'Windows'                # Object of the images to be aligned
base_name = 'Woman'             # Object which gives a good homography
src_path = os.path.join('C:/Users/user/Desktop/Data HS + Mono/', base_name, '/distance285.0mm.png')
dist = np.arange(285, 298, 0.5) # Array of distances of the mono images
os.makedirs('C:/Users/user/Desktop/Data HS + Mono/' + name + '/mono')
os.makedirs('C:/Users/user/Desktop/Data HS + Mono/' + name + '/hs')

for i in dist:
    dst_path = os.path.join(r'C:/Users/user/Desktop/Data HS + Mono/', base_name,'distance', str(i), 'mm.png')
    H, height, width = Find_Homography(src_path, dst_path) # Computes an hpmpgraphy for the chosen object
    dst_path = os.path.join('C:/Users/user/Desktop/Data HS + Mono/', name, 'distance', str(i), 'mm.png') #aligne the disired image
    path_to_save = os.path.join('C:/Users/user/Desktop/Data HS + Mono/', name, 'mono', str(i), 'mm.png')
    Registration_H(dst_path, H, height, width, path_to_save)

#croping HS pictures with the best homography matrix

base_name = 'Poker'          # Object which gives a good homography
src_path = os.path.join('C:/Users/user/Desktop/Data HS + Mono/', base_name, '/distance285.0mm.png')
hs_data = envi.open('C:/Users/user/Desktop/Data HS + Mono/' + base_name +'/HS.hdr','C:/Users/user/Desktop/Data HS + Mono/' + base_name +'/HS.raw')
hs_data_array = np.array(hs_data.load())

# Use the best image to build the homography
best_homography_channel = 75
cv2.imwrite('C:/Users/user/Desktop/Data HS + Mono/' + name +'/hs.png', hs_data_array[0:999, 0:999, best_homography_channel] * 255)
dst_path = 'C:/Users/user/Desktop/Data HS + Mono/' + name +'/hs.png'
H, height, width = Find_Homography(src_path, dst_path)

# Apply the homography on all images
hs_data = envi.open('C:/Users/user/Desktop/Data HS + Mono/' + name +'/HS.hdr','C:/Users/user/Desktop/Data HS + Mono/' + name +'/HS.raw')
hs_data_array = np.array(hs_data.load())
for j in range(150):
    cv2.imwrite('C:/Users/user/Desktop/Data HS + Mono/' + name +'/hs.png', hs_data_array[0:999, 0:999, j] * 255)
    dst_path = 'C:/Users/user/Desktop/Data HS + Mono/' + name + '/hs.png'
    path_to_save = 'C:/Users/user/Desktop/Data HS + Mono/' + name + '/hs/' + str(j) + '.png'
    Registration_H(dst_path, H, height, width, path_to_save)