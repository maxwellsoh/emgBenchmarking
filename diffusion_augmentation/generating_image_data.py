import argparse
import os
import time

import torch

from diffusers import DiffusionPipeline


args = argparse.ArgumentParser()
args.add_argument("--loso_subject_number", type=int, default=5)
args.add_argument("--seed", type=int, default=0)
args.add_argument("--num_images_per_prompt", type=int, default=20)
args.add_argument("--number_of_total_images_per_gesture", type=int, default=1000)
args = args.parse_args()


# start tracking time
start_time = time.time()

# Load your pipeline
pipeline = DiffusionPipeline.from_pretrained(f"examples/text_to_image/custom_models/emg-loso-model_subject-{args.loso_subject_number}",
                                             torch_dtype=torch.float16,
                                             use_safetensors=True).to("cuda")

output_folder = f"examples/dreambooth/emg_images_generated-from-diffusion/emg-loso-model_subject-{args.loso_subject_number}/"
# create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define gestures and guidance scales
gestures = ["Rest", "Flexion", "Abduction", "Extension", "Grip", "Radial_Deviation", "Ulnar_Deviation"]
guidance_scales = [5, 15, 25, 50]

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
