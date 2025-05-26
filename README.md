# Deep Image Restoration: Denoising, Deblurring, and Super-Resolution

This project is a deep learning pipeline for progressive image restoration, developed as part of a Master's-level course on Deep Learning for Image & Video Processing. It integrates multiple tasks including **denoising**, **deblurring**, and **super-resolution** using a sequence of specialized neural architectures: **DnCNN**, **U-Net**, and **EDSR**.

---

## 🚀 Highlights

* 🛠️ **Multi-stage restoration:** Denoising → Deblurring → Super-resolution
* 🏠 **Architectures:**

  * DnCNN for Gaussian noise removal
  * U-Net for motion deblurring with edge enhancement
  * EDSR for 2x image super-resolution
* 🔢 **Metrics:** Achieved up to 30.5 PSNR and 0.928 SSIM on GOPRO-Large
* ⚙️ **Training features:** AMP, ReduceLROnPlateau, MobileNetV3 perceptual loss, Sobel edge loss
* 📃 **Datasets:** GOPRO-Large, Real-World Blur, LSDIR (augmentation)

---

## 💡 Description

The restoration pipeline aims to recover high-quality images from degraded ones (e.g. blurry, noisy, low-res). The final model takes blurry 640x360 images as input and outputs enhanced 1280x720 images through three sequential modules:

1. **DnCNN** for denoising
2. **U-Net** for deblurring (with optional Sobel-based loss)
3. **EDSR** for super-resolution (2x)

These models are trained and fine-tuned independently and jointly. The pipeline uses hybrid losses: MSE + perceptual (MobileNetV3) + edge loss.

---

## 📈 Results

| Model       | PSNR | SSIM  |
| ----------- | ---- | ----- |
| DnCNN       | 27.5 | 0.886 |
| U-Net       | 30.5 | 0.928 |
| EDSR        | 25.4 | 0.858 |
| Final Joint | 24.6 | 0.867 |

Evaluation was performed on the GOPRO-Large test set.

---

## 📚 Datasets

* [GOPRO-Large](https://seungjunnah.github.io/Datasets/gopro) — primary dataset for training and testing
* [Real-World Blur Dataset](http://cg.postech.ac.kr/research/realblur/) — used for deblurring augmentation
* [LSDIR Dataset](https://data.vision.ee.ethz.ch/yawli/) — used for super-resolution data augmentation

---

## 🤖 Implementation Notes

* All models trained on Google Colab (V100 16GB)
* Input size: 640x360 or 360x640
* Optimizer: Adam + ReduceLROnPlateau
* Losses: MSE + 0.1 \* Perceptual + 0.2 \* Edge Loss (final stage)
* AMP: Mixed precision training with GradScaler and autocast

---

## 🔍 Evaluation Metrics

* **PSNR**: Peak Signal-to-Noise Ratio
* **SSIM**: Structural Similarity Index
* **MSE**: Mean Squared Error
* **Edge Loss**: Sobel-based gradient difference
* **Perceptual Loss**: MobileNetV3 feature distance

---

## 🗃️ Pretrained Weights

Weights for the final trained models are available here:
[Google Drive Link](https://drive.google.com/drive/folders/17U7pkEUPILrAtDx19CdSUyG8EUrN6e1c?usp=sharing)

---

## 👤 Author Notes

This project was developed independently as part of a course assignment. Some base model scaffolding was adapted from PyTorch tutorials. Training scripts, perceptual loss implementations, and the full multi-stage pipeline were constructed and tuned by the author.

---

## 🛋️ Disclaimer

This repository currently does not include fine-tuning scripts due to space constraints. Model paths have been removed. See pretrained model link for outputs.
