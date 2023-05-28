![Image Pearer](https://github.com/Sirosky/Image-Pearer/assets/2752448/960a807d-15d7-4f0d-9943-e82abc204986) 
# Image Pearer

Image Pearer automates the creation of image pairs from a LR and a HR source for neural network model training. The primary use case is for pairing images from a DVD to images from a BD, for example. Image Pearer is not intended to be an "all-in-one" tool, but helps eliminate the tedium of manually creating datasets from LR/HR sources.

The instructions assume you have Python, FFMPEG and [ImgAlign](https://github.com/sonic41592/ImgAlign) installed, and an understanding of how to use each.

![Infographic](https://github.com/Sirosky/Image-Pearer/assets/2752448/496c930e-fb99-4bf4-8dd3-efbac47f8955)

***
# How to Use

## 1. Installation
1. Download the [script](https://github.com/Sirosky/Image-Pearer/blob/main/image_pearer.py).
2. Ensure all dependencies are installed: `cv2` and `skimage.metrics`.

## 2. Extract Images
*For Image Pearer to work, you must first extract the images to be paired from the LR and HR source. The below is the recommended approach.*

Use FFMPEG or a similar tool to extract images based on scene detection. For example, `ffmpeg -i "LR.mkv" -filter_complex "select=bitor(gt(scene\,0.4)\,eq(n\,0))" -vsync drop "lr_extracted/%06d.png"`. Be sure to extract images for both the HRs and the LRs. Scene detect is the recommended approach here to ensure that a the image pairs *roughly* match up, even if they're a few images apart. It also decreases the likelihood of false positives resulting in poorly matching pairs.

## 3. Execute
*With the images extracted, you can prepare to run the script.*

1. Review the OPTIONS section at the top of the script to ensure the paths are correct. The default starting value for SSIM and image numbers should be fine, but tweak as needed.
2. Run with `python image_pearer.py`. If successful, you should see the script printing out results and the image pairs created in the output directories.

## 4. ImgAlign (Optional)
*Not always necessary, but for many sources, it's helpful to ensure closely matching pairs.*

[ImgAlign](https://github.com/sonic41592/ImgAlign) is a very helpful tool for making sure image pairs match each other accurately. After creating the pairs with Image Pearer, consider also running ImgAlign to further ensure that they are aligned. ImgAlign is straightforward to use-- simply follow the documentation on the Github page.


