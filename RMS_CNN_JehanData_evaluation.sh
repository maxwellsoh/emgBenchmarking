#for SEED in {0..4}
SEED=4
#do
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=1;
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=2;
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=3;
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=4;
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=5;
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=6;
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=7;
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=8;
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=9;
    python RMS_CNN_JehanData.py --seed=$SEED --leftout_subject=10;
#done

