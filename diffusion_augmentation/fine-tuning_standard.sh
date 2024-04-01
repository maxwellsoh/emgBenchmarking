#!/bin/bash
# Usage: script.sh <MODEL_NAME> <"GESTURE_ARRAY" [comma delimited]> <"SPECIFIC_SUBJECTS" [comma delimited]> \
# <OUTPUT_DIR> <INSTANCE_DIR_GESTURE>

# Standard fine-tuning seems better for many-shot learning and multiple gestures in one model

# Default values for the script arguments
DEFAULT_MODEL_NAME="runwayml/stable-diffusion-v1-5"
DEFAULT_GESTURE_ARRAY="Rest,Extension,Flexion,Ulnar_Deviation,Radial_Deviation,Grip,Abduction"
DEFAULT_SPECIFIC_SUBJECTS="1"
DEFAULT_OUTPUT_DIR="diffusion_augmentation/custom_models/emg-loso-model/cwt_256"
DEFAULT_INSTANCE_DIR_GESTURE="LOSOimages/OzdemirEMG/LOSO_no_scaler_normalization/cwt_256"

# Check for provided script arguments and set defaults if not provided
MODEL_NAME="${1:-$DEFAULT_MODEL_NAME}"
# Convert delimited string to array
IFS=',' read -r -a GESTURE_ARRAY <<< "${2:-$DEFAULT_GESTURE_ARRAY}"
IFS=',' read -r -a SPECIFIC_SUBJECTS <<< "${3:-$DEFAULT_SPECIFIC_SUBJECTS}"
OUTPUT_DIR="${4:-$DEFAULT_OUTPUT_DIR}"
INSTANCE_DIR_GESTURE="${5:-$DEFAULT_INSTANCE_DIR_GESTURE}"

#MODEL_NAME="runwayml/stable-diffusion-v1-5"
#MODEL_NAME="stabilityai/stable-diffusion-2-1"
# MODEL_NAME="ehristoforu/stable-diffusion-v1-5-tiny"

# Use seq to create a sequence. Bash loops use a different syntax.

# Delete corrupted images with size 0
find ${INSTANCE_DIR_GESTURE}/.. -type f -size 0 -exec rm {} +

# Loop through specified subjects only
for i in "${SPECIFIC_SUBJECTS[@]}"; do
  SUBJECT_OUTPUT_DIR="${OUTPUT_DIR}/subject-${i}"

  # Setup folder
  IFS=,; ./diffusion_augmentation/folder_setup.sh ${i} ${INSTANCE_DIR_GESTURE} "${GESTURE_ARRAY[*]}"

  # Use the corrected syntax for variable expansion and command execution
  accelerate launch diffusion_augmentation/train_text_to_image.py \
      --pretrained_model_name_or_path="$MODEL_NAME" \
      --dataset_name="$INSTANCE_DIR_GESTURE/all_data_except" \
      --use_ema \
      --output_dir="$SUBJECT_OUTPUT_DIR" \
      --train_batch_size=1 \
      --gradient_accumulation_steps=2 \
      --learning_rate=1e-5 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --max_train_steps=15000 \
      --max_grad_norm=1 \
      --enable_xformers_memory_efficient_attention \
      --seed=0 \
      --checkpointing_steps=5000 \
      --snr_gamma=5.0

  echo "Done training for subject left out ${i}"
  # Delete temporate folder
  # rm -rf ${INSTANCE_DIR_GESTURE}/all_data_except
done

