#python image_compare_ssim.py

#------------------------#
#OPTIONS
#------------------------#

folder1 = "Output/HR" #Folder containing HR images aligned by ImgAlign
folder2 = "Output/LR" #Folder containing LR images aligned by ImgAlign
folder3 = "Output/LR_low_score_ssim" #Output folders
folder4 = "Output/HR_low_score_ssim"
threshold = 0.6 # Set the threshold for matching score

#------------------------#
#IMPORT LIBRARIES
#------------------------#

import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim # Import SSIM function

#------------------------#
#SCRIPT
#------------------------#

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

        # Calculate the SSIM score between the two images
        # Make sure the images have the same data range
        score = ssim(img1_resized, img2_resized, data_range=255)

        # Append the image name and score to the list if it is less than the threshold
        if score < threshold:
            low_score_images.append((img1, score))

# Print the list of low score images
print("The following images have a SSIM score less than the threshold:")
for img, score in low_score_images:
    print(f"{img}: {score:.2f}")

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