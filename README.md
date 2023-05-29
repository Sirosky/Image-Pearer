![Image Pearer](https://github.com/Sirosky/Image-Pearer/assets/2752448/960a807d-15d7-4f0d-9943-e82abc204986) 
# Image Pearer

Image Pearer automates the creation of image pairs from a LR and a HR source for neural network model training. The primary use case is for pairing images from a DVD to images from a BD, for example. Image Pearer is not intended to be an "all-in-one" tool, but helps eliminate the tedium of manually creating datasets from LR/HR sources.

The instructions assume you have Python, FFMPEG and [ImgAlign](https://github.com/sonic41592/ImgAlign) installed, and an understanding of how to use each.

![Infographic](https://github.com/Sirosky/Image-Pearer/assets/2752448/496c930e-fb99-4bf4-8dd3-efbac47f8955)

***
# How to Use

## 1. Installation
1. Download the repository.
2. Ensure all dependencies are installed: `cv2` and `skimage.metrics`. The rest should come with Python.

## 2. Extract Images
*For Image Pearer to work, you must first extract the images to be paired from the LR and HR source. The below is the recommended approach.*

- Use FFMPEG or a similar tool to extract images based on scene detection. For example, `ffmpeg -i "LR.mkv" -filter_complex "select=bitor(gt(scene\,0.4)\,eq(n\,0))" -vsync drop "lr_extracted/%06d.png"`. Be sure to extract images for both the HRs and the LRs **as an image sequence**. Scene detect is highly recommended to ensure that the image pairs *roughly* match up in the ordering of the LR and HR image sequences. It also decreases the likelihood of false positives resulting in poorly matching pairs.
- When extracting images, a good practice is to make sure the extracted image count the LR and HR images roughly match up. If not, you would have to expand the search range of the script to ensure accurate pairing, which can slow things down drastically.

## 3. Execute
*With the images extracted, you can prepare to run the script.*

1. Review the OPTIONS section at the top of the script to ensure the paths are correct. A good rule of thumb is making sure that the image search range in `num_images` is at least equal to the difference in the HR/LR sequence sizes. You can also do a test run by setting the threshold to a very high value (e.g. 1.0) and seeing what values your dataset is averaging. Then, fine tune the threshold value accordingly.
2. Choose which of the `image_pearer_X` scripts to use. You'll note that there are three of them-- one for each of SSIM, AKAZE and ORB. Each of these three scripts do the same thing, but with different approaches to running the image comparisons. While you should experiment and see what works best for the sourcs you're using, I'd recommend starting with AKAZE.
3. Run the script with `python image_pearer_akaze.py`. If successful, you should see the script printing out results and the image pairs created in the output directories.

## 4. ImgAlign
*Probably the easiest method of making sure the pairs are aligned accurately.*

[ImgAlign](https://github.com/sonic41592/ImgAlign) is a helpful tool for making sure image pairs match each other accurately. For example, oftentimes BDs aren't exactly 2x the resolution of DVDs, which is something ImgAlign can fix. It can also automatically crop and warp as needed to align image pairs. ImgAlign is straightforward to use-- simply follow the documentation on the Github page.

## 5. Double Checking (Optional)

ImgAlign generally does a good job of creating good, matching image pairs. However, you may want to double check the final results with one of the `image_compare_X`. As I generally use AKAZE or ORB for the initial pairing, I like to use SSIM for this final step as unlike AKAZE/ORB, it take a different approach than feature matching so it provides a "second opinion" of sorts. Image pairs with schores that do not meet the prescribed threshold are moved to a separate folder for manual review.

And with that, you should have a dataset!

***
# Credits

- Written with the help of ChatGPT.
- Image Pearer uses SSIM, ORB and AKAZE for image feature matching.
