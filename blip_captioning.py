import pandas as pd 
import os 
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, BlipModel
import time
from tqdm import tqdm
import torch 

begin_time = time.time()

# find all images in /images, 
imgs = os.listdir('downloaded_images')
# keep the jpg without _densepose.jpg
imgs = [i for i in imgs if 'dwt' not in i]
# only keep the png 
imgs = [i for i in imgs if i.endswith('.png')]


df_fn = 'data_blip2.csv'

if os.path.exists(df_fn):
    df = pd.read_csv(df_fn)
else:    
    # create an empty dataframe
    df = pd.DataFrame(columns=['file_name', 'conditioning_image', 'caption'])

# process images in batches of 100

# remove imgs that have already been processed
processed_imgs = df['file_name'].tolist()
imgs = [i for i in imgs if i not in processed_imgs]
print('Number of images to process: ', len(imgs))


STEPSIZE = 250
for i in tqdm(range(0, len(imgs), STEPSIZE)):
    batch_imgs = imgs[i:i+STEPSIZE]

    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    batch_image_paths = ['downloaded_images/' + imagepath for imagepath in batch_imgs]
    
    # fix a porential OSError: image file is truncated error
    try: 
        batch_images = [Image.open(image_path) for image_path in batch_image_paths]
        batch_images = [img.convert('RGB') for img in batch_images]
    except Exception as e:
        print(e)
        print("OS ERROR -- trying to fix")
        
        for image_path in batch_image_paths:
            try:
                img = Image.open(image_path)
                img = img.convert('RGB')
                img.save(image_path)
            except Exception as e:
                print(e)
                # remove the image from the batch
                print("Removing image from batch")
                batch_imgs.remove(image_path.split('/')[-1])
                print(batch_imgs)
                print(image_path.split('/')[-1])
                batch_images = [Image.open(image_path) for image_path in batch_image_paths]
                # delete the image from the folder
                os.remove(image_path)
    
    inputs = processor(images=batch_images, return_tensors="pt", text=["A picture of" for i in range(len(batch_imgs))])



    # GPU acceleration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)  # Move the model to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")


    # Generate captions for the batch
    outputs = model.generate(**inputs, max_new_tokens=100 )
    # print(outputs)
    # captions = processor.decode(model.generate(**processor(images=batch_images, text="A picture of", return_tensors="pt"))[0], skip_special_tokens=True, max_new_tokens=50)
    captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
    print(captions)

    conditioning_images = [imagepath.replace('.png', '_dwt2_thr.png') for imagepath in batch_imgs]
    
    # for imagepath in batch_imgs:
        # caption = processor.decode(model.generate(**processor(images=Image.open('data_folder/train/' + imagepath), text="A picture of", return_tensors="pt", max_new_tokens=50))[0], skip_special_tokens=True)
        # conditioning_image = imagepath.replace('.jpg', '_densepose.jpg')
        
        # batch_captions.append(caption)
        # batch_conditioning_images.append(conditioning_image)
    
    batch_df = pd.DataFrame(list(zip(batch_imgs, conditioning_images, captions)), columns=['file_name', 'conditioning_image', 'caption'])
    df = pd.concat([df, batch_df], ignore_index=True)
    # save dataframe as csv
    
    # SLOW and overwrites the csv
    df.to_csv(df_fn, index=False)
    print('Batch processed and saved to csv. Time elapsed: ', time.time() - begin_time)

print('Done. Time elapsed: ', time.time() - begin_time)
