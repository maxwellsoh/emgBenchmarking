# :fireworks: EMG Benchmarking
EMG gesture classification benchmarking study

# Installation
Install a version of Miniforge distribution `>= Miniforge3-22.3.1-0`, which will give you access to `mamba`. This is a faster and more verbose version of `conda`. 

It is reccomended to run on a Linux x86_64 (amd64) architecture. 

To install the virtual environment, run 
```console
$ mamba env create -n emgbench -f environment.yml
$ pip install -r requirements.txt
$ conda activate emgbench
```

To update the virtual environment, run
```console
$ mamba env update --file enviroment.yml --prune
$ pip install -r requirements.txt
```

FOR DEVELOPMENT: To save the virtual environment, run
```console
$ mamba env export --no-builds > environment.yml
$ pip list --format=freeze > requirements.txt
```


# Benchmarking
To download, organize, and process the publicly available datasets (Ninapro DB2, Ninapro DB5, CapgMyo, Hyser, M (Myoband), Ozdemir, and UCI), run:
```console
$ ./get_datasets.py
```
Jehan's dataset must be requested. 
Or, CNN_EMG will automatically download the necessary datasets for each run. 

# Troubleshooting
If you run into an error, `OSError: [Errno 24] Too many open files`
Run the command 
```console
$ ulimit -n 65536
```
