#!/bin/bash

# Check if a subject number is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <subject_number> <src_dir> <target_dir> <gesture_array>"
  echo "Example: $0 1 examples/dreambooth/emg_images/cwt examples/dreambooth/emg_images/cwt/all_data_except \
  'Rest' 'Extension' 'Flexion' 'Ulnar_Deviation' 'Radial_Deviation' 'Grip' 'Abduction'"
  exit 1
fi

DEFAULT_SUBJECT_NUMBER="1"
DEFAULT_SRC_DIR="examples/dreambooth/emg_images/cwt"
DEFAULT_OUTPUT_DIR="examples/dreambooth/emg_images/cwt/all_data_except"
DEFAULT_GESTURE_ARRAY=('Rest' 'Extension' 'Flexion' 'Ulnar_Deviation' 'Radial_Deviation' 'Grip' 'Abduction')

# Subject number from the first script argument
SUBJECT_NUMBER="$1"
# Base directory for source files
SRC_DIR="$2" # "../examples/dreambooth/emg_images/cwt"
# Base directory for target temporary folders
TARGET_DIR="$3" # "examples/dreambooth/emg_images/cwt/all_data_except"
GESTURE_ARRAY=('Rest' 'Extension' 'Flexion' 'Ulnar_Deviation' 'Radial_Deviation' 'Grip' 'Abduction')

SUBJECT_NUMBER="${1:-$DEFAULT_SUBJECT_NUMBER}"
SRC_DIR="${2:-$DEFAULT_SRC_DIR}"
TARGET_DIR="${3:-$DEFAULT_OUTPUT_DIR}"

# GESTURE_ARRAY=(${2:-${DEFAULT_GESTURE_ARRAY[@]}})
# OUTPUT_DIR="${4:-$DEFAULT_OUTPUT_DIR}"
# SRC_DIR="${5:-$DEFAULT_SRC_DIR}"

# Array of gestures
GESTURE_ARRAY=('Rest' 'Extension' 'Flexion' 'Ulnar_Deviation' 'Radial_Deviation' 'Grip' 'Abduction')

# Initialize metadata string for all gestures
metadata_entries=""

# Loop through each gesture
for gesture in "${GESTURE_ARRAY[@]}"; do
  # Define target directory for "all except subject i"
  TARGET_GESTURE_DIR="${TARGET_DIR}/all_data_except_subject${SUBJECT_NUMBER}_${gesture}"
  # Create the target directory
  mkdir -p "${TARGET_GESTURE_DIR}"

  # Loop through all subject directories to copy the relevant gesture data
  for subject_dir in "${SRC_DIR}"/LOSO_subject*; do
    # Extract subject number
    subject_number=$(basename "$subject_dir" | sed 's/LOSO_subject//')

    # Skip the current subject
    if [ "$subject_number" -ne "$SUBJECT_NUMBER" ]; then
      # Define source gesture directory
      SRC_GESTURE_DIR="${subject_dir}/${gesture}"
      # Check if the source gesture directory exists
      if [ -d "${SRC_GESTURE_DIR}" ]; then
        # Copy the gesture directory from source to target, excluding the current subject
        cp -r "${SRC_GESTURE_DIR}/." "${TARGET_GESTURE_DIR}/"
        echo "Copied ${SRC_GESTURE_DIR} to ${TARGET_GESTURE_DIR}"
      fi
    fi
  done

  # After copying is done for the gesture, generate metadata for copied images
  for image_path in "${TARGET_GESTURE_DIR}"/*.png; do
    image_file=$(basename "$image_path")
    # Append new entry to the metadata string
    metadata_entries+="{\"file_name\": \"all_data_except_subject${SUBJECT_NUMBER}_${gesture}/${image_file}\", \"text\": \"tnu ${gesture} heatmap for loso-cv subject ${SUBJECT_NUMBER}\"}"$'\n'
  done
done

# Write all metadata entries to the file in the all_data_except directory
echo -n "$metadata_entries" > "${TARGET_DIR}/metadata.jsonl"

echo "Folders created and files copied for all except subject ${SUBJECT_NUMBER}. metadata.jsonl file is also created with optimized performance."
