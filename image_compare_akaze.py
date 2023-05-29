#python image_compare_akaze.py

import cv2
import os
import numpy as np
from tqdm import tqdm

# Initialize AKAZE detector and matcher
akaze = cv2.AKAZE_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


# Set the threshold for matching score
threshold = 0.7

# Set the paths of the two folders
folder1 = "Output/HR"
folder2 = "Output/LR"
folder3 = "Output/LR_low_score_akaze"
folder4 = "Output/HR_low_score_akaze"

if not os.path.exists(folder3):
    os.makedirs(folder3)
if not os.path.exists(folder4):
    os.makedirs(folder4)

# Get the list of image names in each folder
images1 = os.listdir(folder1)
images2 = os.listdir(folder2)

# Create an empty list to store the image names and scores
low_score_images = []

# Loop through each image pair
for img1 in tqdm(images1, position=0, leave=True):
    # Check if the image name exists in both folders
    if img1 in images2:
        # Read the images and convert to grayscale
        img1_path = os.path.join(folder1, img1)
        img2_path = os.path.join(folder2, img1)
        img1_gray = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2_gray = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        # Resize the images to the same size
        h1, w1 = img1_gray.shape[:2]
        h2, w2 = img2_gray.shape[:2]
        h = min(h1, h2)
        w = min(w1, w2)
        img1_resized = cv2.resize(img1_gray, (w, h))
        img2_resized = cv2.resize(img2_gray, (w, h))

        # Detect and compute AKAZE keypoints and descriptors
        kp1, des1 = akaze.detectAndCompute(img1_resized, None)
        kp2, des2 = akaze.detectAndCompute(img2_resized, None)
        
        # Filter out the images with no keypoints
        if des1 is None or des2 is None:
            continue
        
        # Convert the descriptors to the same type
        des1 = np.array(des1, dtype=np.uint8)
        des2 = np.array(des2, dtype=np.uint8)

        # Match the descriptors using brute force matcher with Hamming distance
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for match in matches:
            # Check if there are enough values to unpack
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            else:
                # Skip the image pair if not enough matches are found
                continue

        # Calculate the matching score as the ratio of good matches to total keypoints
        # Add a check to avoid dividing by zero
        if len(kp1) > 0 and len(kp2) > 0:
            score = len(good) / max(len(kp1), len(kp2))
        else:
            score = 0
        # print(f"{img1}: {score:.2f}")
        # Append the image name and score to the list if it is less than the threshold
        if score < threshold:
            low_score_images.append((img1, score))

# Print the list of low score images
print(f"The following images have a score less than {threshold}:")
for img, score in low_score_images:
    print(f"{img}: {score:.2f}")

# Create the folders if they do not exist
if not os.path.exists(folder3):
    os.makedirs(folder3)
if not os.path.exists(folder4):
    os.makedirs(folder4)

# Loop through the low score images and move them to the corresponding folders
for img, score in low_score_images:
    # Get the source and destination paths
    src1 = os.path.join(folder1, img)
    src2 = os.path.join(folder2, img)
    dst1 = os.path.join(folder4, img)
    dst2 = os.path.join(folder3, img)

    # Normalize the paths using os.path.normpath
    src1 = os.path.normpath(src1)
    src2 = os.path.normpath(src2)
    dst1 = os.path.normpath(dst1)
    dst2 = os.path.normpath(dst2)

    # Move the files using os.rename
    os.rename(src1, dst1)
    os.rename(src2, dst2)

# Print a message to indicate the completion of the task
print(f"The image pairs with a score falling under {threshold} have been moved to {folder3} and {folder4}.")