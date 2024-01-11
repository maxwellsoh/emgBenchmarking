# emgBenchmarking
EMG gesture classification benchmarking study

# Installation
Install the latest version of miniforge, which will give you access to mamba, 
which is a faster and more verbose version of conda. 

To install the virtual environment, run 
```console
$ conda create -n emgbench
$ conda activate -n emgbench
(emgbench)[emgBenchmarking]$ mamba env create -n new-env-name -f environment.yml```

# Troubleshooting
If you run into an error, `OSError: [Errno 24] Too many open files`
Run the command 
```console
$ ulimit -n 65536
```
