# load metadata.csv
# make sure the image column is image data type
#%%

from datasets import Dataset, Image
import os

os.chdir('downloaded_images')

metadata = Dataset.from_csv("downloaded_images.csv")
image_dataset = metadata.cast_column("file_name", Image())
image_dataset = image_dataset.cast_column("conditioning_image", Image())
image_dataset.push_to_hub('jschoormans/conditioning_haar2_softthresh_filtered')