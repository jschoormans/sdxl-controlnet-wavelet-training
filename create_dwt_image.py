import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from pywt import dwt2, idwt2
from PIL import Image
import os 

# Function for soft thresholding
def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

# Remove the largest largest of the wavelet scales -- set this scale to 0 
def remove_largest_scale(coeffs):
    """
    Remove the largest scale (low-frequency) coefficient from DWT coefficients.
    """
    cA, (cH, cV, cD) = coeffs
    # Replace cA with zeros of the same shape
    cA_zeros = np.zeros_like(cA)
    return (cA_zeros, (cH, cV, cD))


def soft_threshold_max(x, threshold):
    # remove the values above the threshold
    x[np.abs(x) > threshold] = 0
    return x


def run_steps(filename):
# Load and prepare the image
    image = io.imread(filename)
    if image.ndim == 3:
        image = color.rgb2gray(image)

    # Perform 2D DWT
    coeffs = dwt2(image, 'haar')


    coeffs = remove_largest_scale(coeffs)

    cA, (cH, cV, cD) = coeffs


    # Apply soft thresholding (10% of max coefficient value)
    threshold = 0.05 * np.max([np.max(np.abs(cH)), np.max(np.abs(cV)), np.max(np.abs(cD))])
    cH_t = soft_threshold(cH, threshold)
    cV_t = soft_threshold(cV, threshold)
    cD_t = soft_threshold(cD, threshold)
    CA_t = soft_threshold(cA, threshold)


    # Reconstruct the image
    reconstructed = idwt2((CA_t, (cH_t, cV_t, cD_t)), 'haar')
    # convert to PIL 
    reconstructed = Image.fromarray(reconstructed * 255)
    return reconstructed


files = os.listdir('downloaded_images')
files = [f for f in files if f.endswith('.png')]
# exclude the dwt images
files = [f for f in files if not f.endswith('_dwt2_thr.png')]

for file in files:
    try:
        # skip if the file already exists
        if os.path.exists('downloaded_images/' + file.replace('.png', '_dwt2_thr.png')):
            continue
        
        print(file)
        image = Image.open('downloaded_images/' + file)
        # remove transparency
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            image.save('downloaded_images/' + file)

        reconstructed = run_steps('downloaded_images/' + file)
        # convert to RGB, as the original image is RGB
        reconstructed = reconstructed.convert('RGB')
        # hwat is the min and max value of the image
        
        reconstructed.save('downloaded_images/' + file.replace('.png', '_dwt2_thr.png'))
    except: 
        print(f"Error processing {file}")
        continue
