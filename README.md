# :fireworks: EMG Benchmarking
EMG gesture classification benchmarking study

# Installation
Install a version of Miniforge distribution `>= Miniforge3-22.3.1-0`, which will give you access to `mamba`. This is a faster and more verbose version of `conda`. 

To save the virtual environment, run
```console
$ mamba env export --no-builds > environment.yml
$ pip list --format=freeze > requirements.txt
```

To install the virtual environment, run 
```console
$ mamba env create -n emgbench -f environment.yml
$ pip install -r requirements.txt
```

To update the virtual environment, run
```console
$ mamba env update --file enviroment.yml --prune
$ pip install -r requirements.txt
```

To install the forked version of lightning bolts that allows for a pretrained model to be trained with SimCLR:
```console
$ conda remove --force lightning-bolts
$ pip install git+https://github.com/jehanyang/lightning-bolts.git@v0.7
```

If the forked lightning-bolts is updated, then you may need to uninstall and reinstall with:
```console
$ pip uninstall lightning-bolts
$ pip install git+https://github.com/jehanyang/lightning-bolts.git@v0.7
```

# Benchmarking
To get the publicly available datasets Ninapro DB2, Ninapro DB5, CapgMyo, Hyser, M (Myoband), Ozdemir, and UCI, run:
`./get_datasets.py`

This will download, organize, and process the datasets into hdf5 files. 

Jehan's dataset must be requested. 


## Diffusion Augmentation
Go to the `diffusion_augmentation` folder and run the scripts as specified in that `README.md`.
In `CNN_EMG.py` run with argument `--load_diffusion_generated_images=True`. Set `--guidance_scales` if desired.

# Troubleshooting
If you run into an error, `OSError: [Errno 24] Too many open files`
Run the command 
```console
$ ulimit -n 65536
```
