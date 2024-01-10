# image-repair-and-restoration-
This project is an assignment for my master's course in Deep Learning for Image & Video Processing.

In this project, I sequentially utilize DnCNN (denoising), UNet (deblurring), EDSR (super-resolution), and Unsharp Mask sharpening in order to achieve the restoration of blurry images in medium to high resolutions (640x360) to (1280x720), as well as enhancing the clarity of medium-resolution (640x360) images.

The recommended input size for the model is (640x360) or (360x640).

In completing this project, I used Google Colab with the NVIDIA V100 (16GB) hardware accelerator.

## Dataset of Images

The main training for this project is performed using the GOPRO dataset, which can be accessed at https://seungjunnah.github.io/Datasets/gopro.

Additionally, data augmentation is applied during the deblurring phase using the Real-World Blur Dataset, available at http://cg.postech.ac.kr/research/realblur/.

## Methods for evaluating performance
To evaluate the performance of the project's results, used image quality metrics：

&#8226; Peak signal to noise ratio (PSNR),

&#8226; Structural similarity (SSIM) index,

&#8226; Mean squared error (MSE),

&#8226; Sobel operator-based loss functions,

&#8226; Perceptual criterion based MobileNetV3.


## Deep Learning Methods
### Data preprocessing
To resize and normalize images using transforms:

&#8226; Resize the input image to a height of 360 pixels and a width of 640 pixels,

&#8226; Scale the data from [0, 255] to floating-point numbers in the range [0.0, 1.0],

&#8226; Utilize the mean and standard deviation from ImageNet(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

### Optimizer and Learning Rate Scheduler:

&#8226; Use the Adam optimizer for optimizing the model parameters,

&#8226; Implement the ReduceLROnPlateau scheduler to reduce the learning rate based on the validation set loss, allowing for faster detection of convergence during training.

### Automatic Mixed Precision Training:

&#8226; Use GradScaler and autocast are used for automatic mixed precision training to enhance training speed and reduce memory consumption.

### Perceptual criterion

&#8226; Initialize the pre-trained model of MobileNetV3, switch it to evaluation mode, disable gradient computation for all parameters, and pass input images and target images through the feature extraction part of MobileNetV3 to obtain the perceptual loss.

### Early Stopping
To prevent overfitting, stop training when the model doesn't show improvement on the validation set for multiple consecutive epochs.


## Models training process
### DnCNN
The DnCNN is an efficient deep learning model to estimate a residual image from the input image with the Gaussian noise. The underlying noise-free image can be estimated as the difference between the noisy image and the residue image.
