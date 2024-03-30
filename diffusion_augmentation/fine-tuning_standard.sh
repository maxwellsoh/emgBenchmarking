#!/bin/bash

MODEL_NAME="runwayml/stable-diffusion-v1-5"
#MODEL_NAME="stabilityai/stable-diffusion-2-1"

# Bash arrays should be declared without commas and parentheses but with spaces
GESTURE_ARRAY=('Rest' 'Extension' 'Flexion' 'Ulnar_Deviation' 'Radial_Deviation' 'Grip' 'Abduction')

# Use seq to create a sequence. Bash loops use a different syntax.
# Specific subjects to process
SPECIFIC_SUBJECTS=(1 5 9 13 17 21)

# Delete corrupted images with size 0
find ./examples/dreambooth/emg_images/ -type f -size 0 -exec rm {} +

# Loop through specified subjects only
for i in "${SPECIFIC_SUBJECTS[@]}"; do
  # Setup folder
  ./examples/dreambooth/custom_modifications_emg/folder_setup.sh ${i}

  OUTPUT_DIR="examples/text_to_image/custom_models/emg-loso-model_subject-${i}"
  INSTANCE_DIR_GESTURE="examples/dreambooth/emg_images/cwt/all_data_except/"

    # Use the corrected syntax for variable expansion and command execution
  accelerate launch examples/text_to_image/train_text_to_image.py \
      --pretrained_model_name_or_path="$MODEL_NAME" \
      --dataset_name="$INSTANCE_DIR_GESTURE" \
      --use_ema \
      --output_dir="$OUTPUT_DIR" \
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
  rm -rf examples/dreambooth/emg_images/cwt/all_data_except
done

