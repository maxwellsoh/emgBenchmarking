LOSO_CV=$1 # First argument to the script based on whether to run LOSO_CV or not

if [ "$LOSO_CV" = "true" ]; then
    for SEED in {0..4}
    do
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=1
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=2
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=3
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=4
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=5
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=6
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=7
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=8
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=9
        python rawImage_CNN_NinaproDB5.py --epochs=25 --seed=$SEED --leftout_subject=10
    done
else
    for SEED in {0..4}
    do
        python rawImage_CNN_NinaproDB5.py --epochs=200 --seed=$SEED --leftout_subject=0
    done
fi

