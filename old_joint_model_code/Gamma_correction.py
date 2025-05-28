# Use Gamma correction

import os
from PIL import Image
import numpy as np

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")

    if image.mode == 'RGB':
        lut = np.concatenate((lut, lut, lut))

    return image.point(lut)


input_folder = '/content/drive/MyDrive/image_output/denoised'
output_folder = '/content/drive/MyDrive/image_output/gamma_denoised'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set gamma value
gamma_value = 0.7

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        corrected_img = gamma_correction(img, gamma_value)

        # Save image
        corrected_img.save(os.path.join(output_folder, filename))
