# emgBenchmarking
EMG gesture classification benchmarking study

# Installation
Install a version of Miniforge distribution `>= Miniforge3-22.3.1-0`, which will give you access to `mamba`. This is a faster and more verbose version of `conda`. 

To install the virtual environment, run 
```console
$ mamba env create -n emgbench -f environment.yml
```
# Benchmarking
Currently, to download the datasets, you will have to go to the website that hosts the datasets, download each individuals data (for Ninapro DB2, DB5, and Capgmyo), downolad the whole dataset (for Ozdemir's open dataset), or request the dataset (for Jehan's dataset). 

You will follow this up with running the python notebooks `dataset_processing_[DATASET-NAME].ipynb` in order to process the dataset into hdf5 files (with the exception of Jehan's dataset, which is already in hdf5 format). 

# Troubleshooting
If you run into an error, `OSError: [Errno 24] Too many open files`
Run the command 
```console
$ ulimit -n 65536
```
