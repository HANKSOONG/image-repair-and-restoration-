# image-repair-and-restoration-
This project is an assignment for my master's course in Deep Learning for Image & Video Processing.

In this project, we sequentially utilize DnCNN (denoising), UNet (deblurring), EDSR (super-resolution), and Unsharp Mask sharpening in order to achieve the restoration of blurry images in medium to high resolutions (640x360) to (1280x720), as well as enhancing the clarity of medium-resolution (640x360) images.

## Dataset of Images

The main training for this project is performed using the GOPRO dataset, which can be accessed at https://seungjunnah.github.io/Datasets/gopro.

Additionally, data augmentation is applied during the deblurring phase using the Real-World Blur Dataset, available at http://cg.postech.ac.kr/research/realblur/.
****

