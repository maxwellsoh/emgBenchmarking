# :fireworks: EMG Benchmarking
EMG gesture classification benchmarking study

# Installation
To use emgbench, you will need to install a version of Miniforge distribution `>= Miniforge3-22.3.1-0`, which will give you access to `mamba`. This is a faster and more verbose version of `conda`. It is reccomended that you run on a Linux x86_64 (amd64) architecture. 

To install the virtual environment, you can run:
```console
$ mamba env create -n emgbench -f environment.yml
```
use pip to install the required packages:
```
$ pip install -r requirements.txt
```
and then activate your environment: 
```
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
To download, organize, and process the publicly available datasets (Ninapro DB2, Ninapro DB5, CapgMyo, Hyser, M (Myoband), Ozdemir, and UCI), you can run:
```console
$ ./get_datasets.py
```
Alternatively, CNN_EMG will automatically download any missing datasets for each run. 
Note that Jehan's dataset must be requested. 

# Running
Examples of config files are available in ./config/example.yaml. 

To use a preset configuration, run:
```console
$ ./run_config.py --preset {name}
```

To use a custom configuration, in the form ('''for i {1..max}; run CNN_EMG with field = $i'''), run: 
```console
$ .run_config.py --custom --config {config_file} --fields {fields} --max {i}
```

To use the command line to pass in arguments, run: 
```console
$ ./CNN_EMG --argument
```

# Troubleshooting
If you run into an error, `OSError: [Errno 24] Too many open files`
Run the command 
```console
$ ulimit -n 65536
```




