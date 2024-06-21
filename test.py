import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from pywt import dwt2, idwt2

# Load and prepare the image
image = io.imread('test.jpg')
if image.ndim == 3:
    image = color.rgb2gray(image)

# Perform 2D DWT
coeffs = dwt2(image, 'haar')

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




coeffs = remove_largest_scale(coeffs)

cA, (cH, cV, cD) = coeffs


# Apply soft thresholding (10% of max coefficient value)
threshold = 0.1 * np.max([np.max(np.abs(cH)), np.max(np.abs(cV)), np.max(np.abs(cD))])
cH_t = soft_threshold(cH, threshold)
cV_t = soft_threshold(cV, threshold)
cD_t = soft_threshold(cD, threshold)
CA_t = soft_threshold(cA, threshold)




# Reconstruct the image
reconstructed = idwt2((CA_t, (cH_t, cV_t, cD_t)), 'haar')

# Prepare DWT coefficients for visualization
def prepare_coeffs(coeffs):
    return np.vstack((
        np.hstack((coeffs[0], coeffs[1][0])),
        np.hstack((coeffs[1][1], coeffs[1][2]))
    ))

original_dwt = prepare_coeffs(coeffs)
thresholded_dwt = prepare_coeffs((CA_t, (cH_t, cV_t, cD_t)))


# Plotting
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')

axs[0, 1].imshow(np.log(np.abs(original_dwt) + 1), cmap='gray')
axs[0, 1].set_title('Original DWT (log scale)')

axs[1, 0].imshow(np.log(np.abs(thresholded_dwt) + 1), cmap='gray')
axs[1, 0].set_title('Thresholded DWT (log scale)')

axs[1, 1].imshow(reconstructed, cmap='gray')
axs[1, 1].set_title('Reconstructed Image')

for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
plt.savefig('output.png')
