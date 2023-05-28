#python image_pearer.py

# Import libraries
import cv2
import skimage.metrics
import os
import shutil

# Options
lr_path = "lr_extracted" # Path to LR folder
hr_path = "hr_extracted" # Path to HR folder
output_lr_path = "LR" # Path to output LR folder
output_hr_path = "HR" # Path to output HR folder
ssim_threshold = 0.9 # Define SSIM threshold
num_images = 5 # Define number of images to check in either direction
prepend_string = "" # Define string to prepend to image name

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

# Loop through each image in LR folder
for lr_index, lr_image in enumerate(lr_images):
    # Read LR image
    lr_image_path = os.path.join(lr_path, lr_image)
    lr_image_data = cv2.imread(lr_image_path)

    # Convert LR image to grayscale
    lr_image_data = cv2.cvtColor(lr_image_data, cv2.COLOR_BGR2GRAY)

    # Get the corresponding HR image name
    hr_image = lr_image

    # Get the list of image names in HR folder
    hr_images = os.listdir(hr_path)

    # Try to get the index of the HR image in the HR folder
    try:
        hr_index = hr_images.index(hr_image)
    except ValueError:
        # If not found, use the index of the LR image instead
        hr_index = lr_index

    # Initialize the best SSIM score and image name
    best_ssim = 0
    best_hr_image = None

    # Loop through num_images before and after the HR image
    for i in range(-num_images, num_images + 1):
        # Skip if the index is out of range
        if hr_index + i < 0 or hr_index + i >= len(hr_images):
            continue

        # Read HR image
        hr_image_path = os.path.join(hr_path, hr_images[hr_index + i])
        hr_image_data = cv2.imread(hr_image_path)

        # Resize HR image to match LR image
        hr_image_data = cv2.resize(hr_image_data, (lr_image_data.shape[1], lr_image_data.shape[0]))

        # Convert HR image to grayscale
        hr_image_data = cv2.cvtColor(hr_image_data, cv2.COLOR_BGR2GRAY)

        # Compute SSIM
        ssim = skimage.metrics.structural_similarity(lr_image_data, hr_image_data)

        # Update the best SSIM score and image name if higher than current best
        if ssim > best_ssim:
            best_ssim = ssim
            best_hr_image = hr_images[hr_index + i]

    # Check if the best SSIM score is at least ssim_threshold
    if best_ssim >= ssim_threshold:
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

            # Print a message that a successful image pair was created with LR and HR image names and SSIM score
            print(f"Created a successful image pair with {new_lr_image} and {new_hr_image} with SSIM: {best_ssim}")

            # Increment the total image pairs created by 1
            total_pairs += 1

            # Increment the output image index by 1
            output_index += 1

# Print a message with the total image pairs created at the end
print(f"Total image pairs created: {total_pairs}")
