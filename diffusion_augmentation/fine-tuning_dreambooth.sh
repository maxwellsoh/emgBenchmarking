#!/bin/bash
# DreamBooth seems better for few-shot learning than standard fine-tuning

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

  for j in "${!GESTURE_ARRAY[@]}"; do  # Iterate over array indices
    OUTPUT_DIR="examples/dreambooth/custom_models/emg-loso-model_subject-${i}_${GESTURE_ARRAY[$j]}"
    INSTANCE_DIR_GESTURE="examples/dreambooth/emg_images/cwt/all_data_except/all_data_except_subject${i}_${GESTURE_ARRAY[$j]}"

    # Use the corrected syntax for variable expansion and command execution
    accelerate launch examples/dreambooth/train_dreambooth.py \
      --pretrained_model_name_or_path="$MODEL_NAME" \
      --instance_data_dir="$INSTANCE_DIR_GESTURE" \
      --output_dir="$OUTPUT_DIR" \
      --instance_prompt="tnu ${GESTURE_ARRAY[$j]} heatmap for loso-cv subject ${i}" \
      --train_batch_size=1 \
      --gradient_accumulation_steps=2 \
      --learning_rate=5e-6 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --max_train_steps=400 \
      --enable_xformers_memory_efficient_attention \
      --seed=0 \
      --checkpointing_steps=5000

    echo "Done training gesture ${GESTURE_ARRAY[$j]} for subject left out ${i}"
  done
  # Delete temporate folder
  rm -rf examples/dreambooth/emg_images/cwt/all_data_except
done

