# image-repair-and-restoration-
This project is an assignment for my master's course in Deep Learning for Image & Video Processing.

In this project, I sequentially utilize DnCNN (denoising), UNet (deblurring), EDSR (super-resolution) in order to achieve the restoration of blurry images in medium to high resolutions (640x360) to (1280x720), as well as enhancing the clarity of medium-resolution (640x360) images.

The recommended input size for the model is (640x360) or (360x640).

In completing this project, I used Google Colab with the NVIDIA V100 (16GB) hardware accelerator.

The final model, when tested on the GOPRO\_Large datasets test set, achieved an average PSNR (Peak Signal-to-Noise Ratio) of 28.957791 and an average SSIM (Structural Similarity Index Measure) of 0.737406352.

## Dataset of Images

The main training for this project is performed using the GOPRO\_Large dataset, which can be accessed at https://seungjunnah.github.io/Datasets/gopro.

Data augmentation is applied during the deblurring phase using the Real-World Blur Dataset, available at http://cg.postech.ac.kr/research/realblur/.

Data augmentation is applied during the deblurring phase using the LSDIR Dataset, available at https://data.vision.ee.ethz.ch/yawli/.

## Code
The initial code for the section 'Define and initialize the neural network' comes from PyTorch: 
https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html.

The fine-tuning code of the model differs from the training code only in the part where the pre-trained model is loaded. Therefore, I have not uploaded the fine-tuning code.

## Model
The pre-trained weights of the model can be obtained from https://drive.google.com/drive/folders/17U7pkEUPILrAtDx19CdSUyG8EUrN6e1c?usp=sharing.

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
### 1. DnCNN (denoising)
The DnCNN is an efficient deep learning model to estimate a residual image from the input image with the Gaussian noise. The underlying noise-free image can be estimated as the difference between the noisy image and the residue image.

1. Following the typical definition of the DnCNN denoising model, all images were resized to (360x640) dimensions. An initial learning rate of 0.0001 was set, and the blurry images from the GOPRO dataset training set were processed. The mean squared error (MSE) weight was set to 1, and the perceptual loss weight to 0.1. The total loss function was defined as the sum of MSE and perceptual loss, each multiplied by their respective weights. After training for 48 epochs, the initial pre-trained model was obtained.

2. After obtaining the initial pre-trained model, fine-tuning was performed by incorporating gamma-corrected blurry images from the GOPRO dataset into the training process. After training for 66 epochs, the second version of the denoising model was achieved.

3. Building upon the second version of the denoising model, all images were resized to (640x360) dimensions, and gamma-corrected blurry images from the GOPRO dataset were included in the fine-tuning process. After training for 58 epochs, the final denoising model was obtained, with training loss of 0.0971 and validation loss of 0.0939. At this point, the average PSNR (Peak Signal-to-Noise Ratio) between the denoised and sharp images was 27.5, and the average SSIM (Structural Similarity Index) was 0.886.

### 2. U-Net (deblurring)
U-Net is a convolutional neural network that was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg. The network is based on a fully convolutional neural network whose architecture was modified and extended to work with fewer training images and to yield more precise segmentation. U-Net is particularly well-suited for image deblurring tasks because it can combine low-level and high-level feature information and preserve details through skip connections, which is crucial for recovering clear image content.

1. Iterating through all the blurry images in the GOPRO training dataset, standardize and resize them to (360x640) dimensions to use as input for the pre-trained DnCNN model, resulting in denoised images. These denoised images are then input into the pre-trained U-Net model. With an MSE weight of 1 and a perceptual loss weight of 0.1, the total loss function is defined as the sum of MSE and perceptual loss, each multiplied by their respective weights. After training for 77 epochs, the initial pre-trained model is obtained.

2. After obtaining the initial pre-trained model, the denoised images were gamma-corrected with a gamma value set to 0.7. Simultaneously, the Real-World Blur Dataset was introduced, incorporating denoised images, gamma-corrected denoised images, and images from the Real-World Blur Dataset. After training for 82 epochs, the second version of the denoising model was achieved.

3. Resize the original images to (640x360) dimensions and input them into the DnCNN model to obtain denoised images. Building upon the second version of the denoising model, fine-tune the U-Net model, and after training for 73 epochs, the third version of the model is obtained.

4. It was observed that the edge effects in the output of the third version of the model were not satisfactory. To address this issue, Sobel operator-based loss functions were introduced with a weight of 1 and added to the total loss function. After an additional 22 epochs of fine-tuning, the final deblurred images were obtained. At this point, the training loss was 0.4154, the validation loss was 0.3835, the average PSNR (Peak Signal-to-Noise Ratio) between the deblurred and sharp images was 30.5, and the average SSIM (Structural Similarity Index) was 0.928.

### 3. EDSR (super-resolution)
EDSR is an enhanced deep residual network for single-image super-resolution, widely used for super-resolution tasks. 

1.  Due to GPU performance limitations and image size constraints, I simply input the deblurred images obtained from U-Net, which are (640x360) in size, into EDSR to train a super-resolution model that increases the image resolution by a factor of two to (1280x720). With an MSE weight of 1 and a perceptual loss weight of 0.1, the total loss function is defined as the sum of MSE and perceptual loss, each multiplied by their respective weights. After training for 94 epochs, the initial pre-trained model is obtained with a training loss of 0.1240, a validation loss of 0.1320, an average PSNR (Peak Signal-to-Noise Ratio) of 25.4, and an average SSIM (Structural Similarity Index) of 0.858.

2.  Although the EDSR model performed well on the initial training data, I found it to be unsatisfactory when applied to actual images I downloaded. Therefore, I used the first 10,000 images from the LSDIR Dataset for data augmentation. After 30 epochs, the average PSNR (Peak Signal-to-Noise Ratio) reached 24.9, and the average SSIM (Structural Similarity Index) was 0.867.

### 4. joint model
After obtaining the three models, I employed a joint training approach. With an MSE weight of 1, a perceptual loss weight of 0.1, and an edge loss weight of 0.2, the total loss function was defined as the sum of MSE, perceptual loss, and edge loss, each multiplied by their respective weights. After training for 10 epochs, the final model was achieved, with a training loss of 0.2358, a validation loss of 0.1736, an average PSNR (Peak Signal-to-Noise Ratio) between deblurred and sharp images of 24.629592931317745, and an average SSIM (Structural Similarity Index) of 0.8670229522462696.
