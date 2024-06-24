# load metadata.csv
# make sure the image column is image data type
#%%

from datasets import Dataset, Image
import os


# fix some issues



metadata = Dataset.from_csv("data_blip2.csv")
# remove the image with a file name that is too long
metadata = metadata.filter(lambda x: len(x['file_name']) < 200)

os.chdir('downloaded_images')
image_dataset = metadata.cast_column("file_name", Image())
image_dataset = image_dataset.cast_column("conditioning_image", Image())
image_dataset.push_to_hub('jschoormans/conditioning_haar2_softthresh_filtered')