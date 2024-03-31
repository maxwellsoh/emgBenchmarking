import argparse
import zarr
import torch
import os
import re

def load_images(subject_number, guidance_scales, gesture_names=None):
    # Construct the base path for the images
    base_path = f"../diffusers/examples/dreambooth/emg-zarr_generated-from-diffusion/cwt/LOSO_subject{subject_number}"
    
    gesture_to_number = {key: value for value, key in enumerate(gesture_names)}

    # Initialize a list to hold the loaded images and labels
    loaded_image_groups = []
    labels = []
                
    # Open the Zarr group to explore available datasets
    if os.path.exists(base_path):
        
        z_group = zarr.open_group(base_path, mode="r")
        
        # Use regular expression to find matches for gesture types and scales
        pattern = re.compile(r"(.+)_guidance_scale-(\d+)")
        
        for key in z_group.array_keys():
            match = pattern.match(key)
            if match:
                gesture, scale = match.groups()
                img_array = None
                if scale in guidance_scales:  
                    img_array = z_group[key][:]
                    # Convert numpy array to PyTorch tensor
                    img_tensor = torch.tensor(img_array)
                    loaded_image_groups.append(img_tensor)
                    # Append the correct number label to the labels list
                    # labels.append(f"{gesture}_{scale}")
                    label_numbers = gesture_to_number[gesture]*torch.ones(img_tensor.shape[0], dtype=torch.long)
                    # Encode into one-hot
                    label_numbers = torch.nn.functional.one_hot(label_numbers, num_classes=len(gesture_to_number))
                    labels.append(label_numbers)
                if img_array is not None:
                    print(f"Loaded {key}, which has shape {img_array.shape}")
    
    return loaded_image_groups, labels

def main():
    parser = argparse.ArgumentParser(description='Load Zarr images into PyTorch.')
    parser.add_argument('--loso_subject_number', type=str, help='LOSO subject number', default="1")
    parser.add_argument('--guidance_scales', type=str, help='Guidance scale to load', default="5,15,25,50")
    parser.add_argument('--dataset', type=str, help='Dataset to load', default="Ozdemir")
    
    args = parser.parse_args()
    
    gesture_names = None
    if args.dataset == "Ozdemir":
        import utils_OzdemirEMG as utils
        gesture_names = utils.gesture_labels_partial

    image_groups, labels = load_images(args.loso_subject_number, args.guidance_scales.split(","), gesture_names)
    total_number_of_images = sum([len(img_group) for img_group in image_groups])
    print(f"Loaded {len(image_groups)} image datasets, which have a total of {total_number_of_images} images.")

if __name__ == "__main__":
    main()
