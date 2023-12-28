for SEED in {0..4}
do
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=1
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=2
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=3
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=4
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=5
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=6
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=7
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=8
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=9
    python rawImage_CNN_NinaproDB5.py --seed=$SEED --leftout_subject=10
done

