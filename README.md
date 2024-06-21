## Controlnet DWT training 
## 21 Juni

CONDA ENV: torch_xformers

1. Download_data.py 



Training directly (otherwise conda is fucked)

python train_controlnet_sdxl.py  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  --output_dir="model_out"  --dataset_name=jschoormans/conditioning_haar2_softthresh_filtered  --conditioning_image_column=conditioning_image  --image_column=file_name  --caption_column=text  --resolution=512  --learning_rate=1e-5  --validation_image "validation_images/image_003_dwt.png" "validation_images/image_011_dwt.png"  --validation_prompt "Girl smiling, professional dslr photograph, dark background, studio lights, high quality" "Portrait of a runner"  --train_batch_size=2  --num_train_epochs=3  --tracker_project_name="controlnet"  --enable_xformers_memory_efficient_attention  --checkpointing_steps=5000  --validation_steps=1000  --report_to wandb  --push_to_hub --variant=fp16



python train_controlnet.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
 --output_dir="model_out" \
 --dataset_name=jschoormans/conditioning_haar2_softthresh_filtered \
 --conditioning_image_column=conditioning_image \
 --image_column=file_name \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "validation_images/image_003_dwt.png" "validation_images/image_011_dwt.png"  \
 --validation_prompt "Women's Converse Chuck Taylor All Star Madison Floral Lined Sneakers" "Person posing"  \
 --num_validation_images=1 \
 --train_batch_size=4 \
 --num_train_epochs=3 \
 --tracker_project_name="controlnet-wavelet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=250 \
 --validation_steps=50 \
 --mixed_precision=fp16 \
 --report_to "wandb" \
 --push_to_hub \
 --logging_dir="logs"
