# from PIL import Image
# import os 

# val_dir = "validation_images"

# for f in os.listdir(val_dir):
#     load = Image.open(f'{val_dir}/{f}')
#     load = load.resize((512, 512))
#     load.save(f'{val_dir}/{f}')

## 
# check the image sizes in downloaded_images and validation_images

import os
from PIL import Image
images = os.listdir('downloaded_images')
images.sort()
images = [i for i in images if i.endswith('.png')]
for i in images:
    
    # if the image file name contains 'dwt2_thr' then delete it
    if 'dwt2_thr' in i:
        os.remove(f'downloaded_images/{i}')
        continue
    
    img = Image.open(f'downloaded_images/{i}')
    if img.size[0] > 1024 or img.size[1] > 1024:
        print(f"Image {i} has size {img.size}")
    else:
        continue
    
    print("RESIZING IMAGE...")
    
    # resize to 1024x1024 MAXIMUM long side
    if img.size[0] > img.size[1]:
        img = img.resize((1024, int(1024 * img.size[1] / img.size[0])))
    else:
        img = img.resize((int(1024 * img.size[0] / img.size[1]), 1024))
    img.save(f'downloaded_images/{i}')    
    