# emgBenchmarking
EMG gesture classification benchmarking study

# Installation
To install the virtual environment, run 
```console
$ conda create -n emgbench
$ conda activate -n emgbench
(emgbench)[emgBenchmarking]$ pip install -r requirements.txt
```

# Troubleshooting
If you run into an error, `OSError: [Errno 24] Too many open files`
Run the command 
```console
$ ulimit -n 65536
```
