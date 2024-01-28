# emgBenchmarking
EMG gesture classification benchmarking study

# Installation
Install a version of Miniforge distribution `>= Miniforge3-22.3.1-0`, which will give you access to `mamba`. This is a faster and more verbose version of `conda`. 

To install the virtual environment, run 
```console
$ mamba env create -n emgbench -f environment.yml
$ pip install -r requirements.txt
```

# Troubleshooting
If you run into an error, `OSError: [Errno 24] Too many open files`
Run the command 
```console
$ ulimit -n 65536
```
