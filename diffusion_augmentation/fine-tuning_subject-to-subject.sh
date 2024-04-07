#!/bin/bash
# Usage: script.sh <MODEL_NAME> <"GESTURE_ARRAY" [comma delimited]> <"SPECIFIC_SUBJECTS" [comma delimited]> \
# <OUTPUT_DIR> <INSTANCE_DIR_GESTURE>

# Standard fine-tuning seems better for many-shot learning and multiple gestures in one model

# Default values for the script arguments
DEFAULT_MODEL_NAME="runwayml/stable-diffusion-v1-5"
DEFAULT_GESTURE_ARRAY="Rest,Extension,Flexion,Ulnar_Deviation,Radial_Deviation,Grip,Abduction"
DEFAULT_SPECIFIC_SUBJECTS="1,5,9,13,17,21"
DEFAULT_OUTPUT_DIR="diffusion_augmentation/custom_models/OzdemirEMG/cwt/subject-to-subject"
DEFAULT_INSTANCE_DIR_GESTURE="LOSOimages/OzdemirEMG/LOSO_no_scaler_normalization/cwt"
DEFAULT_SUBJECT_TO_MAP_TO="8"

# Check for provided script arguments and set defaults if not provided
MODEL_NAME="${1:-$DEFAULT_MODEL_NAME}"
# Convert delimited string to array
IFS=',' read -r -a GESTURE_ARRAY <<< "${2:-$DEFAULT_GESTURE_ARRAY}"
IFS=',' read -r -a SPECIFIC_SUBJECTS <<< "${3:-$DEFAULT_SPECIFIC_SUBJECTS}"
OUTPUT_DIR="${4:-$DEFAULT_OUTPUT_DIR}"
INSTANCE_DIR_GESTURE="${5:-$DEFAULT_INSTANCE_DIR_GESTURE}"
SUBJECT_TO_MAP_TO="${6:-$DEFAULT_SUBJECT_TO_MAP_TO}"

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
  IFS=,; ./diffusion_augmentation/folder_setup_for_fine-tuning_subject-to-subject.sh ${i} ${INSTANCE_DIR_GESTURE} ${SUBJECT_TO_MAP_TO} "${GESTURE_ARRAY[*]}"

  # Use the corrected syntax for variable expansion and command execution
  accelerate launch diffusion_augmentation/train_controlnet.py \
      --pretrained_model_name_or_path=$MODEL_NAME  \
      --output_dir=$SUBJECT_OUTPUT_DIR  \
      --dataset_name="$INSTANCE_DIR_GESTURE/all_data_except"  \
      --learning_rate=1e-5  \
      --resolution=224 \
      --validation_image "${INSTANCE_DIR_GESTURE}/LOSO_subject${i}/Abduction/image_2880_subject${i}.png" "${INSTANCE_DIR_GESTURE}/LOSO_subject${i}/Extension/image_480_subject${i}.png"  \
      "${INSTANCE_DIR_GESTURE}/LOSO_subject${i}/Flexion/image_960_subject${i}.png" "${INSTANCE_DIR_GESTURE}/LOSO_subject${i}/Grip/image_2400_subject${i}.png" \
      "${INSTANCE_DIR_GESTURE}/LOSO_subject${i}/Radial_Deviation/image_1920_subject${i}.png" "${INSTANCE_DIR_GESTURE}/LOSO_subject${i}/Rest/image_0_subject${i}.png" \
      "${INSTANCE_DIR_GESTURE}/LOSO_subject${i}/Ulnar_Deviation/image_1440_subject${i}.png" \
      "${INSTANCE_DIR_GESTURE}/LOSO_subject0/Abduction/image_2880_subject0.png" "${INSTANCE_DIR_GESTURE}/LOSO_subject0/Extension/image_480_subject0.png"  \
      "${INSTANCE_DIR_GESTURE}/LOSO_subject0/Radial_Deviation/image_1920_subject0.png" "${INSTANCE_DIR_GESTURE}/LOSO_subject0/Rest/image_0_subject0.png" \
      "${INSTANCE_DIR_GESTURE}/LOSO_subject0/Flexion/image_960_subject0.png" "${INSTANCE_DIR_GESTURE}/LOSO_subject0/Grip/image_2400_subject0.png" \
      "${INSTANCE_DIR_GESTURE}/LOSO_subject0/Ulnar_Deviation/image_1440_subject0.png" \
      --validation_prompt "heatmap transformation for loso-cv subject" "heatmap transformation for loso-cv subject"  \
      "heatmap transformation for loso-cv subject" "heatmap transformation for loso-cv subject" \
      "heatmap transformation for loso-cv subject" "heatmap transformation for loso-cv subject" \
      "heatmap transformation for loso-cv subject" \
      "heatmap transformation for loso-cv subject" "heatmap transformation for loso-cv subject"  \
      "heatmap transformation for loso-cv subject" "heatmap transformation for loso-cv subject" \
      "heatmap transformation for loso-cv subject" "heatmap transformation for loso-cv subject" \
      "heatmap transformation for loso-cv subject" \
      --train_batch_size=3 \
      --gradient_accumulation_steps=2 \
      --gradient_checkpointing \
      --use_8bit_adam \
      --seed=0 \
      --checkpointing_steps=1000 \
      --checkpoints_total_limit=3  \
      --report_to=wandb \
      --validation_steps=500 \
      --image_column="target_image" \
      --lr_scheduler="cosine_with_restarts" \
      --num_train_epochs=1

  echo "Done training for subject left out ${i}"
  # Delete temporate folder
  rm -rf ${INSTANCE_DIR_GESTURE}/all_data_except
done

