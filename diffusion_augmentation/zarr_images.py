import argparse
import os

import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm


# Setup argument parser
args = argparse.ArgumentParser()
args.add_argument("--loso_subject_number", type=int, default=1)
args.add_argument("--gesture_labels", type=str, default="Rest,Extension,Flexion,Ulnar_Deviation,Radial_Deviation,Grip,Abduction")
args.add_argument("--guidance_scales", type=str, default="5,15,25,50")
args.add_argument("--image_directory", type=str, default="LOSOimages_generated-from-diffusion/OzdemirEMG/cwt/")
args.add_argument("--output_directory", type=str, default="LOSOimages_zarr_generated-from-diffusion/OzdemirEMG/cwt/")
args.add_argument("--subject_folder_suffix", type=str, default="")
args.add_argument("--nested_folder", type=bool, default=False)
args = args.parse_args()

# Configure paths and constants
MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def normalize(images):
    return (images - MEAN) / STD

def load_and_process_image(image_path):
    img = Image.open(image_path)
    img_array = np.asarray(img).astype(np.float32) / 255  # Using float32 for a balance between precision and memory usage
    return img_array

def process_subject(gesture_labels, guidance_scales):
    save_dir = f'{args.output_directory}subject-{args.loso_subject_number}{args.subject_folder_suffix}/'
    os.makedirs(save_dir, exist_ok=True)

    zarr_group = zarr.open_group(save_dir, mode='w')

    parent_image_folder = os.path.join(args.image_directory, f'subject-{args.loso_subject_number}{args.subject_folder_suffix}')

    for gesture_label in gesture_labels:
        for guidance_scale in guidance_scales:
            images_dir = f'{parent_image_folder}/gesture-{gesture_label}/'
            os.makedirs(images_dir, exist_ok=True)
            
            image_files = [f for f in os.listdir(images_dir) if f'guidance-scale-{guidance_scale}_' in f]
            print(f'Processing {len(image_files)} images for gesture {gesture_label} at guidance scale {guidance_scale}')
            with tqdm(total=len(image_files)) as pbar:
                # Preallocate array for performance if images are uniform in size
                sample_image_path = os.path.join(images_dir, image_files[0])
                sample_image_array = load_and_process_image(sample_image_path)
                images_shape = (len(image_files), *sample_image_array.shape)
                images_dtype = sample_image_array.dtype
                images_data = np.zeros(images_shape, dtype=images_dtype)

                for i, image_file in enumerate(image_files):
                    image_path = os.path.join(images_dir, image_file)
                    img_array = load_and_process_image(image_path)
                    images_data[i] = normalize(img_array)

                    pbar.update(1)

                # Write to Zarr in one go to minimize disk I/O
                dataset_name = f'{gesture_label}_guidance_scale-{guidance_scale}'
                zarr_group.array(dataset_name, images_data, chunks=(1, *images_data.shape[1:]))

def process_subject_in_nested_folders(gesture_labels, guidance_scales):
    save_dir = f'{args.output_directory}subject-{args.loso_subject_number}{args.subject_folder_suffix}/'
    os.makedirs(save_dir, exist_ok=True)

    zarr_group = zarr.open_group(save_dir, mode='w')

    parent_image_folder = os.path.join(args.image_directory, f'subject-{args.loso_subject_number}{args.subject_folder_suffix}')
    print("Looking in folders in", parent_image_folder)
    nested_subject_image_folders = [f'{parent_image_folder}/{folder}' for folder in os.listdir(parent_image_folder)]
    for nested_image_directory in nested_subject_image_folders:
        # Make zarr sub-group for nested folder
        nested_folder_name = nested_image_directory.split('/')[-1]
        nested_zarr_group = zarr_group.create_group(nested_folder_name)
        for gesture_label in gesture_labels:
            for guidance_scale in guidance_scales:
                images_dir = f'{nested_image_directory}/gesture-{gesture_label}/'
                os.makedirs(images_dir, exist_ok=True)
                
                image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
                print(f'Processing {len(image_files)} images for gesture {gesture_label} at guidance scale {guidance_scale}')
                with tqdm(total=len(image_files)) as pbar:
                    # Preallocate array for performance if images are uniform in size
                    sample_image_path = os.path.join(images_dir, image_files[0])
                    sample_image_array = load_and_process_image(sample_image_path)
                    images_shape = (len(image_files), *sample_image_array.shape)
                    images_dtype = sample_image_array.dtype
                    images_data = np.zeros(images_shape, dtype=images_dtype)

                    for i, image_file in enumerate(image_files):
                        image_path = os.path.join(images_dir, image_file)
                        img_array = load_and_process_image(image_path)
                        images_data[i] = normalize(img_array)

                        pbar.update(1)

                    # Write to Zarr in one go to minimize disk I/O
                    dataset_name = f'{gesture_label}_guidance_scale-{guidance_scale}'
                    nested_zarr_group.array(dataset_name, images_data, chunks=(1, *images_data.shape[1:]))

def main():
    gesture_labels = args.gesture_labels.split(',')
    guidance_scales = args.guidance_scales.split(',')
    if args.nested_folder:
        process_subject_in_nested_folders(gesture_labels, guidance_scales)
    else:
        process_subject(gesture_labels, guidance_scales)

if __name__ == "__main__":
    main()
