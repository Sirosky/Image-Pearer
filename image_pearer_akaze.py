#python image_pearer_akaze.py

#------------------------#
#IMPORT LIBRARIES
#------------------------#
import cv2
import os
import shutil

#------------------------#
#OPTIONS
#------------------------#

lr_path = "lr_extracted" # Path to LR folder
hr_path = "hr_extracted" # Path to HR folder
output_lr_path = "LR" # Path to output LR folder
output_hr_path = "HR" # Path to output HR folder
match_threshold = 0.25 # Define match threshold. Up to 1.0. Higher = stricter requirements to create an image pair. You may have to adjust depending on the source.
num_images = 25 # Define number of images to check in either direction. Increase to search further for an image to pair with. This might be necessary if scene detect is generating vastly different results between the LR and HR sources. Note that increasing the value decreases speed.
prepend_string = "akaze_" # Define a string to prepend to image name. For example, putting "blah" would result in output images such as "blah_000001.png".
resize_height = 720 # Define a fixed height to resize the images
resize_width = 1280 # Define a fixed width to resize the images

distance_modifier = .75 #Suggest leaving this alone in most scenarios

#------------------------#
#SCRIPT
#------------------------#

# Check if output_lr_path and output_hr_path exist and create them if they don't
if not os.path.exists(output_lr_path):
    os.makedirs(output_lr_path)
if not os.path.exists(output_hr_path):
    os.makedirs(output_hr_path)

# Get the list of image names in LR folder
lr_images = os.listdir(lr_path)

# Initialize the total image pairs created
total_pairs = 0

# Initialize the output image index
output_index = 1

# Create an AKAZE object with default parameters
akaze = cv2.AKAZE_create()

# Create an empty list to store the blacklisted HR images
blacklist = []

# Loop through each image in LR folder
for lr_index, lr_image in enumerate(lr_images):
    
    # Read LR image
    lr_image_path = os.path.join(lr_path, lr_image)
    lr_image_data = cv2.imread(lr_image_path)
    # print(f"Current index {lr_index}")
    
    # Convert LR image to grayscale
    lr_image_data = cv2.cvtColor(lr_image_data, cv2.COLOR_BGR2GRAY)

    # Resize the LR image to the fixed height and width
    lr_image_data = cv2.resize(lr_image_data, (resize_width, resize_height))

    # Compute AKAZE keypoints and descriptors for LR image
    lr_kp, lr_des = akaze.detectAndCompute(lr_image_data, None)

    # Get the corresponding HR image name
    hr_image = lr_image

    # Get the list of image names in HR folder
    hr_images = os.listdir(hr_path)
    # print(f"Debug: Current lr_image {lr_image}. Current hr_image {hr_image}.")

    # Try to get the index of the HR image in the HR folder
    try:
        hr_index = hr_images.index(hr_image)
    except ValueError:
        # If not found, use the index of the LR image instead
        # print(f"Debug: HR image {hr_index} not found. Switching to {lr_index}.")
        hr_index = lr_index

    # Initialize the best match ratio and image name
    best_match_ratio = 0
    best_hr_image = None

    # Loop through num_images before and after the HR image
    for i in range(-num_images, num_images + 1):
        # Skip if the index is out of range or the image is blacklisted
        if hr_index + i < 0 or hr_index + i >= len(hr_images) or hr_images[hr_index + i] in blacklist:
            continue

        # Read HR image
        hr_image_path = os.path.join(hr_path, hr_images[hr_index + i])
        hr_image_data = cv2.imread(hr_image_path)

        # Convert HR image to grayscale
        hr_image_data = cv2.cvtColor(hr_image_data, cv2.COLOR_BGR2GRAY)

        # Resize the HR image to the fixed height and width
        hr_image_data = cv2.resize(hr_image_data, (resize_width, resize_height))

        # Compute AKAZE keypoints and descriptors for HR image
        hr_kp, hr_des = akaze.detectAndCompute(hr_image_data, None)

        # Check if the HR image has any keypoints or descriptors
        if hr_kp and hr_des is not None:
            # Create a brute force matcher object with default parameters
            bf = cv2.BFMatcher()

            # Try to match the descriptors of LR and HR images using brute force matcher
            try:
                matches = bf.knnMatch(lr_des, hr_des, k=2)
            except cv2.error:
                # If the descriptors have different types or columns, print a message and move onto the next image
                # print(f"Descriptors have different types or columns for {lr_image} and {hr_images[hr_index + i]}.")
                continue

            # Apply ratio test to filter out good matches
            good_matches = []
            for match in matches:
                # Check if match has two values
                if len(match) == 2:
                    # Unpack the values
                    m, n = match
                    # Apply ratio test
                    if m.distance < distance_modifier * n.distance:
                        good_matches.append(m)

            # Try to compute the match ratio as the number of good matches divided by the number of LR keypoints
            try:
                match_ratio = len(good_matches) / len(lr_kp)
            except ZeroDivisionError:
                # If AKAZE failed to locate keypoints for the LR image, print a message and move onto the next image
                # print(f"AKAZE failed to locate keypoints for {lr_image}.")
                continue

            # Update the best match ratio and image name if higher than current best
            if match_ratio > best_match_ratio:
                best_match_ratio = match_ratio
                best_hr_image = hr_images[hr_index + i]
        else:
            # If the HR image has no keypoints or descriptors, add it to the blacklist and print a message
            blacklist.append(hr_images[hr_index + i])
            print(f"AKAZE failed to locate keypoints for {hr_path}/{hr_images[hr_index + i]}. Added to blacklist.")

    # Check if the best match ratio is at least match_threshold
    if best_match_ratio >= match_threshold:
        # Get the extension of the LR image name
        lr_ext = os.path.splitext(lr_image)[1]

        # Generate a new LR image name with prepend string and output index with leading zeros and fixed width of 6 digits
        new_lr_image = prepend_string + str(output_index).zfill(6) + lr_ext

        # Copy the LR image to output LR folder with new name
        output_lr_image_path = os.path.join(output_lr_path, new_lr_image)
        shutil.copy(lr_image_path, output_lr_image_path)

        # Check if the best HR image exists in HR folder
        if best_hr_image in hr_images:
            # Get the extension of the best HR image name
            hr_ext = os.path.splitext(best_hr_image)[1]

            # Generate a new HR image name with prepend string and output index with leading zeros and fixed width of 6 digits
            new_hr_image = prepend_string + str(output_index).zfill(6) + hr_ext

            # Copy the best HR image to output HR folder with new name as LR image
            best_hr_image_path = os.path.join(hr_path, best_hr_image)
            output_hr_image_path = os.path.join(output_hr_path, new_hr_image)
            shutil.copy(best_hr_image_path, output_hr_image_path)

            # Print a message that a successful image pair was created with LR and HR image names and match ratio
            print(f"Created a successful image pair with {lr_path}/{lr_image} and {hr_path}/{best_hr_image} with match ratio of {best_match_ratio} as {new_hr_image}.")

            # Increment the total image pairs created by 1
            total_pairs += 1

        # Increment the output image index by 1
        output_index += 1
    else:
        print(f"Did not locate a pair for {lr_path}/{lr_image}. Highest match ratio was {best_match_ratio} from {hr_path}/{best_hr_image}.")

# Print a message with the total image pairs created at the end
print(f"Total image pairs created: {total_pairs}")
print(f"AKAZE was unable to create keypoints for the following images from {hr_path}, and excluded them for pairing: {blacklist}")
