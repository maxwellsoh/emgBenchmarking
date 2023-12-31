#!/bin/bash

print_usage() {
    printf "Usage: script [OPTIONS]\n"
    printf "Options:\n"
    printf " -l  Specify whether to run leave-one-out cross-validation (True or False). Default is False.\n"
    printf " -v  Specify whether to run k-fold cross-validation (True or False). Default is False.\n"
    printf " -k  Specify the number of folds if using k-fold cross-validation. Default is 5.\n"
    printf " -s  Set the seed for reproducibility. Default is 0.\n"
    printf " -e  Set the number of epochs to train for. Default is 25.\n"
    printf " -c  Specify whether to use cyclical learning rate (True or False). Default is False.\n"
    printf " -a  Specify whether to use cosine annealing with warm restarts (True or False). Default is False.\n"
    printf " -i  Specify what index fold to start at. Default is 1.\n"
    printf " -r  Specify whether to use RMS normalization (True or False). Default is False.\n"
    printf " -n  Specify the number of windows to use for RMS normalization. Should be an even factor of sample timesteps (Probably 50). Default is 10.\n"
    printf " -m  Specify whether to use a magnitude image concatenation (True or False). Default is False.\n"
    printf " -h  Show usage information.\n"
}

# Initialize variables with default values
LOSO_CV="False"
K_FOLDS_ON="False"
K_FOLDS=5
SEED=0
EPOCHS=25
FOLD_INDEX_START=1
CYCLICAL_LR_ON="False"
COSINE_ANNEALING_ON="False"
RMS_ON="False"
RMS_WINDOWS=10
MAGNITUDE_ON="False"
MODEL_TO_USE="resnet50"

# Parse command line arguments
while getopts 'l:v:k:s:e:c:a:i:r:n:m:o:h' flag; do  # Changed flags to single letters
    case "${flag}" in
        l) LOSO_CV="${OPTARG}" ;;
        v) K_FOLDS_ON="${OPTARG}" ;;
        k) K_FOLDS="${OPTARG}" ;;
        s) SEED="${OPTARG}" ;;
        e) EPOCHS="${OPTARG}" ;;
        c) CYCLICAL_LR_ON="${OPTARG}" ;;  # -clr changed to -c
        a) COSINE_ANNEALING_ON="${OPTARG}" ;;  # -ca changed to -a
        i) FOLD_INDEX_START="${OPTARG}" ;;
        r) RMS_ON="${OPTARG}" ;;
        n) RMS_WINDOWS="${OPTARG}" ;;
        m) MAGNITUDE_ON="${OPTARG}" ;;
        o) MODEL_TO_USE="${OPTARG}" ;;
        h) print_usage
           exit 0 ;;
        *) print_usage
           exit 1 ;;
    esac
done

echo "*******************************************"
echo "Running with settings:"
echo "LOSO_CV: $LOSO_CV"
echo "K_FOLDS_ON: $K_FOLDS_ON"
echo "K_FOLDS: $K_FOLDS"
echo "SEED: $SEED"
echo "EPOCHS: $EPOCHS"
echo "CYCLICAL_LR_ON: $CYCLICAL_LR_ON"
echo "COSINE_ANNEALING_ON: $COSINE_ANNEALING_ON"
echo "FOLD_INDEX_START: $FOLD_INDEX_START"
echo "RMS_ON: $RMS_ON"
echo "RMS_WINDOWS: $RMS_WINDOWS"
echo "MAGNITUDE_ON: $MAGNITUDE_ON"
echo "MODEL_TO_USE: $MODEL_TO_USE"
echo "*******************************************"

# Main conditional execution based on LOSO_CV and K_FOLDS_ON flags
if [ "$LOSO_CV" = "True" ]; then
    echo "Running Leave-One-Subject-Out Cross-Validation..."
    for SUBJECT in {1..10}
    do
        python rawImage_CNN_NinaproDB5.py --epochs=$EPOCHS --seed=$SEED --leftout_subject=$SUBJECT --turn_on_cyclical_lr=$CYCLICAL_LR_ON --turn_on_cosine_annealing=$COSINE_ANNEALING_ON --turn_on_rms=$RMS_ON --num_rms_windows=$RMS_WINDOWS --turn_on_magnitude=$MAGNITUDE_ON --model=$MODEL_TO_USE
    done
elif [ "$K_FOLDS_ON" = "True" ]; then
    echo "Running stratified k-folds cross validation..."
    for FOLD_INDEX in $(seq $FOLD_INDEX_START $K_FOLDS)
    do
        python rawImage_CNN_NinaproDB5.py --epochs=$EPOCHS --seed=$SEED --leftout_subject=0 --turn_on_kfold=True --kfold=$K_FOLDS --fold_index=$FOLD_INDEX --turn_on_cyclical_lr=$CYCLICAL_LR_ON --turn_on_cosine_annealing=$COSINE_ANNEALING_ON --turn_on_rms=$RMS_ON --num_rms_windows=$RMS_WINDOWS --turn_on_magnitude=$MAGNITUDE_ON --model=$MODEL_TO_USE
    done
else
    echo "Running standard training..."
    echo "Value of cosine_annealing"
    echo $COSINE_ANNEALING_ON
    echo "Value of cyclical_lr"
    echo $CYCLICAL_LR_ON
    python rawImage_CNN_NinaproDB5.py --epochs=$EPOCHS --seed=$SEED --leftout_subject=0 --turn_on_cyclical_lr=$CYCLICAL_LR_ON --turn_on_cosine_annealing=$COSINE_ANNEALING_ON --turn_on_rms=$RMS_ON --num_rms_windows=$RMS_WINDOWS --turn_on_magnitude=$MAGNITUDE_ON --model=$MODEL_TO_USE
fi
