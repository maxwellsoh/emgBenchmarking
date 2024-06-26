#!/bin/bash

# Check if a subject number is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <subject_number> <src_dir> <gesture_array>"
  echo "Example: $0 1 examples/dreambooth/emg_images/cwt \
  'Rest,Extension,Flexion,Ulnar_Deviation,Radial_Deviation,Grip,Abduction'"
  exit 1
fi

# Default values
DEFAULT_SUBJECT_NUMBER="1"
DEFAULT_SRC_DIR="LOSOimages/OzdemirEMG/LOSO_no_scaler_normalization/cwt_256"
DEFAULT_GESTURE_ARRAY="Rest,Extension,Flexion,Ulnar_Deviation,Radial_Deviation,Grip,Abduction"

# Processing inputs
SUBJECT_NUMBER="${1:-$DEFAULT_SUBJECT_NUMBER}"
SRC_DIR="${2:-$DEFAULT_SRC_DIR}"
# Convert delimited string to array for GESTURE_ARRAY
IFS=',' read -r -a GESTURE_ARRAY <<< "${3:-$DEFAULT_GESTURE_ARRAY}"
# Initialize metadata string for all gestures
metadata_entries=""
OUTPUT_DIR="${SRC_DIR}/all_data_except"

for gesture in "${GESTURE_ARRAY[@]}"; do
  echo "Current Gesture: $gesture"
  # Define target directory for "all except subject i"
  OUTPUT_GESTURE_DIR="${OUTPUT_DIR}/all_data_except_subject${SUBJECT_NUMBER}_${gesture}"
  echo "Output Gesture Directory: $OUTPUT_GESTURE_DIR"
  # Create the target directory
  mkdir -p "$OUTPUT_GESTURE_DIR"

  echo "Source Directory: $SRC_DIR"
  # Loop through all subject directories to copy the relevant gesture data
  for subject_dir in "${SRC_DIR}"/LOSO_subject*; do
    # Extract subject number
    subject_number=$(basename "$subject_dir" | sed 's/LOSO_subject//')

    echo "Current Subject: $subject_number"
    # Skip the current subject
    if [ "$subject_number" -ne "$SUBJECT_NUMBER" ]; then
      # Define source gesture directory
      SRC_GESTURE_DIR="${subject_dir}/${gesture}"
      # Check if the source gesture directory exists
      if [ -d "${SRC_GESTURE_DIR}" ]; then
        # Copy the gesture directory from source to target, excluding the current subject
        cp -r "${SRC_GESTURE_DIR}/." "${OUTPUT_GESTURE_DIR}/"
        echo "Copied ${SRC_GESTURE_DIR} to ${OUTPUT_GESTURE_DIR}"
      fi
    fi
  done

  # After copying is done for the gesture, generate metadata for copied images
  for image_path in "${OUTPUT_GESTURE_DIR}"/*.png; do
    image_file=$(basename "$image_path")
    # Append new entry to the metadata string
    metadata_entries+="{\"file_name\": \"all_data_except_subject${SUBJECT_NUMBER}_${gesture}/${image_file}\", \"text\": \"tnu ${gesture} heatmap for loso-cv subject ${SUBJECT_NUMBER}\"}"$'\n'
  done
done

# Write all metadata entries to the file in the all_data_except directory
echo -n "$metadata_entries" > "${OUTPUT_DIR}/metadata.jsonl"

echo "Folders created and files copied for all except subject ${SUBJECT_NUMBER}. metadata.jsonl file is also created with optimized performance."
