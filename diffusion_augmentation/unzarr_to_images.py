import multiprocessing
import os

import argparse
import numpy as np
import zarr
from PIL import Image


args = argparse.ArgumentParser()
args.add_argument("--zarr_path", type=str, default="LOSOimages_zarr/OzdemirEMG/LOSO_no_scaler_normalization/cwt/")
args.add_argument("--save_dir", type=str,  default="LOSOimages/OzdemirEMG/LOSO_no_scaler_normalization/cwt/")
args.add_argument("--gesture_labels", type=str, default="Rest,Extension,Flexion,Ulnar_Deviation,Radial_Deviation,Grip,Abduction")
args.add_argument("--total_subjects", type=int, default=40)
args = args.parse_args()

def denormalize(images):
    # Define mean and std from imageNet
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

    # Ensure images is a numpy array
    images = np.array(images)

    # Denormalize
    images = images * std + mean

    # Clip the values to ensure they are within [0,1] as expected for image data
    images = np.clip(images, 0, 1)

    return images

# Get labels for the gestures
def getLabels ():
    numGestures = 7 # Could also be 10 if doing full dataset
    timesteps_for_one_gesture = 480
    labels = np.zeros((timesteps_for_one_gesture*numGestures))
    for i in range(timesteps_for_one_gesture):
        for j in range(numGestures):
            labels[j * timesteps_for_one_gesture + i] = j
    return labels

# Define a function to process a single image
def process_image(image_data, labels, gesture_labels_partial, save_dir, i, j):
    # Transpose the image array from (3, 224, 224) to (224, 224, 3)
    img_array = np.transpose(image_data, (1, 2, 0))

    # Denormalize the image
    img_array = denormalize(img_array)

    # If your data is in the range [0, 1], scale it to [0, 255] and convert to uint8
    if img_array.dtype == np.float64:
        img_array = (img_array * 255).astype(np.uint8)

    # Convert the numpy array to a PIL Image
    img = Image.fromarray(img_array)

    # Get the gesture label for the current image
    gesture_label = gesture_labels_partial[int(labels[j])]

    # Construct the folder path for the gesture
    gesture_folder = os.path.join(save_dir, gesture_label)

    # Make the gesture folder if it doesn't exist
    os.makedirs(gesture_folder, exist_ok=True)

    # Construct the filename for the image within the gesture folder
    filename = f'{gesture_folder}/image_{j}_subject{i}.png'

    # Save the image
    img.save(filename)

    # Optional: Print status
    if (j + 1) % 100 == 0:
        print(f'Saved {j + 1} images for subject {i}')

# Define a function to process a range of images
def process_images_range(start, end):
    for i in range(start, end):
        # Path to the Zarr dataset
        zarr_path = args.zarr_path + f"LOSO_subject{i}/"
        save_dir = args.save_dir + f"LOSO_subject{i}/"
        gesture_labels = args.gesture_labels.split(',')

        # Make save_dir if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Load the Zarr array
        z_array = zarr.open(zarr_path, mode='r')

        # Assuming the array is 2D, directly convert it to a numpy array
        # For multidimensional data, you might need to select a specific slice or reduce dimensions
        image_data = np.array(z_array)

        # Get the labels for the gestures
        labels = getLabels()

        # Iterate through each image
        for j, img_array in enumerate(image_data):
            process_image(img_array, labels, gesture_labels, save_dir, i, j)

# Define the number of processes to use
num_processes = multiprocessing.cpu_count()

# Create a list of process ranges
process_ranges = []
total_number_of_subjects=args.total_subjects
number_of_subjects_per_process, remainder = divmod(total_number_of_subjects, num_processes)

start = 0
end = 0
for i in range(num_processes):
    start = end
    end = start + number_of_subjects_per_process
    # This remainder is used to distribute the remaining subjects to the first few processes
    if remainder > 0:
        end += 1
        remainder -= 1

    process_ranges.append((start, end))

print("Process ranges:", process_ranges)

# Create a pool of processes
pool = multiprocessing.Pool(processes=num_processes)

# Map the process_ranges to the process_images_range function
pool.starmap(process_images_range, process_ranges)

# Close the pool
pool.close()
pool.join()
