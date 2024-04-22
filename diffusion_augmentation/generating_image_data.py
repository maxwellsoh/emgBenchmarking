import argparse
import os
import time

import torch

from diffusers import DiffusionPipeline


args = argparse.ArgumentParser()
args.add_argument("--loso_subject_number", type=int, default=1)
args.add_argument("--seed", type=int, default=0)
args.add_argument("--num_images_per_prompt", type=int, default=20)
args.add_argument("--number_of_total_images_per_gesture", type=int, default=1000)
args.add_argument("--pretrained_model_path", type=str, default="diffusion_augmentation/custom_models/OzdemirEMG/cwt_256/")
args.add_argument("--output_base_folder", type=str, default="LOSOimages_generated-from-diffusion/")
args.add_argument("--gesture_labels", type=str, default="Rest,Extension,Flexion,Ulnar_Deviation,Radial_Deviation,Grip,Abduction")
args.add_argument("--guidance_scales", type=str, default="5,15,25,50")
args = args.parse_args()


# start tracking time
start_time = time.time()

# Load your pipeline
pipeline = DiffusionPipeline.from_pretrained(f"{args.pretrained_model_path}subject-{args.loso_subject_number}",
                                             torch_dtype=torch.float16,
                                             use_safetensors=True).to("cuda")

# Define the output folder
specific_folder = '/'.join(args.pretrained_model_path.split('/')[2:])
output_folder = f"{args.output_base_folder}{specific_folder}subject-{args.loso_subject_number}/"
# create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define gestures and guidance scales
gestures = args.gesture_labels.split(',')
guidance_scales = [int(i) for i in args.guidance_scales.split(',')]

NUMBER_IMAGES_PER_PROMPT = args.num_images_per_prompt
NUMBER_IMAGES_PER_GESTURE = args.number_of_total_images_per_gesture

assert NUMBER_IMAGES_PER_GESTURE % NUMBER_IMAGES_PER_PROMPT == 0, "NUMBER_IMAGES_PER_GESTURE must be divisible by NUMBER_IMAGES_PER_PROMPT"

for scale in guidance_scales:

    for gesture in gestures:
        for i in range(NUMBER_IMAGES_PER_GESTURE // NUMBER_IMAGES_PER_PROMPT):
            # Generate the image
            images = pipeline(f"tnu {gesture} heatmap for loso-cv subject {args.loso_subject_number}",
                             num_inference_steps=50,
                             guidance_scale=scale,
                             num_images_per_prompt=NUMBER_IMAGES_PER_PROMPT,
                             seed=args.seed).images

            for j, image in enumerate(images):
                image = image.resize((224,224))
                image_number = i*NUMBER_IMAGES_PER_PROMPT + j

                specific_output_folder = os.path.join(output_folder, f"gesture-{gesture}")
                os.makedirs(specific_output_folder, exist_ok=True)

                image.save(os.path.join(specific_output_folder, f"guidance-scale-{scale}_gesture-{gesture}_image-{image_number}.png"))

                current_time = time.time()
                delta_time = current_time - start_time
                print(f"Saved image {image_number} for gesture {gesture} with guidance scale {scale} at time {delta_time:.2f} seconds")
