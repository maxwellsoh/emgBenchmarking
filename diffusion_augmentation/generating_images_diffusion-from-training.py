import argparse
import os
import time

import torch
import torchvision

from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


args = argparse.ArgumentParser()
args.add_argument("--loso_subject_number", type=int, default=1)
args.add_argument("--seed", type=int, default=0)
args.add_argument("--batch_size", type=int, default=40)
args.add_argument("--pretrained_model_path", type=str, default="diffusion_augmentation/custom_models/OzdemirEMG/cwt/")
args.add_argument("--output_base_folder", type=str, default="LOSOimages_generated-from-diffusion/")
args.add_argument("--training_images_folder", type=str, default="LOSOimages/OzdemirEMG/LOSO_no_scaler_normalization/cwt/")
args.add_argument("--gesture_labels", type=str, default="Rest,Extension,Flexion,Ulnar_Deviation,Radial_Deviation,Grip,Abduction")
args.add_argument("--guidance_scales", type=str, default="0")
args = args.parse_args()


# start tracking time
start_time = time.time()

# Load your pipeline
pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(f"{args.pretrained_model_path}subject-{args.loso_subject_number}",
                                             torch_dtype=torch.float16,
                                             use_safetensors=True).to("cuda")

# Define the output folder
specific_output_folder = '/'.join(args.pretrained_model_path.split('/')[2:])
output_folder = f"{args.output_base_folder}{specific_output_folder}subject-{args.loso_subject_number}_training-img2img/"
# create folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define gestures and guidance scales
gestures = args.gesture_labels.split(',')
guidance_scales = [int(i) for i in args.guidance_scales.split(',')]

# list all training folders for subjects other than the loso subject
training_images_subject_folders = [f"{args.training_images_folder}LOSO_subject{i}/" for i in range(0, 40) if i != args.loso_subject_number]

for training_images_subject_folder in training_images_subject_folders:
    subject_output_folder = os.path.join(output_folder, f"subject-{training_images_subject_folder.split('/')[-2]}")
    for scale in guidance_scales:
        for gesture in gestures:
            # Load the training images in the folder as PIL images
            for gesture_folder in os.listdir(training_images_subject_folder):
                if gesture_folder == gesture:
                    training_images = [Image.open(os.path.join(training_images_subject_folder, gesture_folder, image_name)) for image_name in os.listdir(os.path.join(training_images_subject_folder, gesture_folder))]
                    
            training_images_tensor = torch.stack([torchvision.transforms.ToTensor()(image) for image in training_images]).to("cuda")
            number_images_for_gesture = len(training_images_tensor)
                
            assert number_images_for_gesture % args.batch_size == 0, "Number of images for gesture must be divisible by batch size"
            
            for i in range(number_images_for_gesture // args.batch_size):
                
                prompt_list = [f"tnu heatmap for loso-cv subject {args.loso_subject_number}"] * args.batch_size
                # Generate the image
                images = pipeline(prompt = prompt_list,
                                image=training_images_tensor[i*args.batch_size:(i+1)*args.batch_size],
                                num_inference_steps=50,
                                guidance_scale=scale,
                                strength=0.35,
                                #  num_images_per_prompt=NUMBER_IMAGES_PER_PROMPT,
                                seed=args.seed).images

                for j, image in enumerate(images):
                    # if image is all black, regenerate image
                    while not image.getbbox():
                        images = pipeline(prompt = prompt_list[0],
                                        image=training_images_tensor[i*args.batch_size+j],
                                        num_inference_steps=50,
                                        guidance_scale=scale,
                                        strength=0.35,
                                        #  num_images_per_prompt=NUMBER_IMAGES_PER_PROMPT,
                                        seed=args.seed).images
                        image = images[0]
                        
                    image = image.resize((224,224))
                    image_number = i*args.batch_size + j

                    specific_output_folder = os.path.join(subject_output_folder, f"gesture-{gesture}")
                    os.makedirs(specific_output_folder, exist_ok=True)

                    image.save(os.path.join(specific_output_folder, f"guidance-scale-{scale}_gesture-{gesture}_image-{image_number}.png"))

                    current_time = time.time()
                    delta_time = current_time - start_time
                    print(f"Saved image {image_number} for gesture {gesture} with guidance scale {scale} at time {delta_time:.2f} seconds")
