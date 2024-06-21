import os
import asyncio
import aiohttp
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
import csv
from io import BytesIO
import json
import itertools

# Load the dataset in streaming mode
dataset = load_dataset("Thouph/Laion_aesthetics_5plus_1024_33M_csv", split="train", streaming=True)
N_TOTAL_IMAGES = 2500
# Create a directory to store the downloaded images
output_dir = "downloaded_images"
os.makedirs(output_dir, exist_ok=True)

# CSV and progress file paths
csv_file = os.path.join(output_dir, "downloaded_images.csv")
progress_file = os.path.join(output_dir, "progress.json")

# Create CSV header if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "text", "conditioning_image"])

# Load progress if it exists
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    start_index = progress['last_processed_index'] + 1
else:
    start_index = 0

# Set of existing files
existing_files = set(os.listdir(output_dir))

# Asynchronous download function
async def download_and_save_image(session, item, i):
    filename = f"image_{i:03d}.png"
    if filename in existing_files:
        print(f"Skipping existing image {i}")
        return None

    image_url = item['URL']

    try:
        async with session.get(image_url, timeout=30) as response:
            if response.status == 200:
                image_data = await response.read()
                image = Image.open(BytesIO(image_data))
                filepath = os.path.join(output_dir, filename)
                image.save(filepath)
                
                condition_image = filename.replace('.png', '_dwt2_thr.png')
                text = item['TEXT']
                
                return [filename, text, condition_image]
            else:
                print(f"Failed to download image {i}: HTTP status {response.status}")
    except Exception as e:
        print(f"Error downloading image {i}: {e}")
    
    return None

async def process_chunk(chunk, session, start_i):
    tasks = [download_and_save_image(session, item, i) for i, item in enumerate(chunk, start=start_i)]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

async def main():
    chunk_size = 50
    total_images = N_TOTAL_IMAGES
    downloaded_images = len([f for f in os.listdir(output_dir) if f.endswith('.png')])

    async with aiohttp.ClientSession() as session:
        pbar = tqdm(total=total_images, initial=downloaded_images, desc="Processing images")
        
        start_index = 0
        
        dataset_iter = iter(dataset.skip(start_index))
        while downloaded_images < total_images:
            chunk = list(itertools.islice(dataset_iter, chunk_size))
            if not chunk:
                print("Reached end of dataset before downloading 2000 images")
                break

            results = await process_chunk(chunk, session, downloaded_images)
            
            # Write results to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(results)
            
            # Update progress
            downloaded_images += len(results)
            with open(progress_file, 'w') as f:
                json.dump({'last_processed_index': start_index + downloaded_images - 1}, f)
            
            pbar.update(len(results))
            start_index += len(chunk)
        
        pbar.close()

if __name__ == "__main__":
    asyncio.run(main())

print("Image processing complete!")
