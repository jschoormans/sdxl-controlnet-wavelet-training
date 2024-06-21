# Activate the conda environment
# Activate the conda environment
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust this path if your Anaconda installation is different
conda activate torch_xformers

# print current environment
echo "Current environment:"
conda env list
# print pip list 
echo "Pip list:"
pip list

accelerate launch train_controlnet_sdxl.py \
 --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
 --output_dir="model_out" \
 --dataset_name=jschoormans/conditioning_haar2_softthresh_filtered \
 --conditioning_image_column=conditioning_image \
 --image_column=file_name \
 --caption_column=caption \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "validation_images/image_003_dwt.png" "validation_images/image_011_dwt.png" \
 --validation_prompt "Girl smiling, professional dslr photograph, dark background, studio lights, high quality" "Portrait of a runner" \
 --train_batch_size=4 \
 --num_train_epochs=3 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=500 \
 --validation_steps=100 \
 --report_to wandb \
 --push_to_hub \
 # --gradient_accumulation_steps=4 \
 # --gradient_checkpointing \
 # --use_8bit_adam \
 # --set_grads_to_none



