from PIL import Image
import os 

val_dir = "validation_images"

for f in os.listdir(val_dir):
    load = Image.open(f'{val_dir}/{f}')
    load = load.resize((512, 512))
    load.save(f'{val_dir}/{f}')