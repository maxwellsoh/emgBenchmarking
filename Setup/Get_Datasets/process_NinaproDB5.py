# %%
# extract original and restimulus data from ./NinaproDB5/sX/sX_e2_A1.mat
import pandas as pd
#from sklearn import preprocessing, model_selection
#from scipy.signal import butter,filtfilt,iirnotch,hilbert
#from PyEMD import EMD
#import wandb
import scipy.io as sio
from tqdm import tqdm
import os
import argparse

# %%
# import ./NinaproDB5/sX/sX_e2_A1.mat for X from 1 to 10
# extract original data and restimulus data

# original data: emg, goniometer, acc, force
# restimulus data: restimulus, restimulus_pos, restimulus_vel, restimulus_acc

# TODO: turn into python script rather than keeping as python notebook
# TODO: add optional arguments for which exercises to load, and other parrameters as needed

# Add optional arguments for which exercises to load
# Create the parser
# parser = argparse.ArgumentParser(description="Include arguments for loading different data files")

# Add argument for exercises to load
# parser.add_argument('--exercises', type=int, nargs="+", help='List the exercises of the 3 to load. The most popular for benchmarking seem to be 2 and 3. Can format as \'--exercises 2 3\'', default=2)

# args = parser.parse_args()

# manually assgn args.exercises to a list of exercises to load for now to test python notebook
class args():
    exercises = [1,2,3]

# TODO: rechunk this so it's not just one big function. I just wasn't sure which ones
def get():
    data = {}
    emg_microvolts = {}
    emg_class = {}

    for subject in range(1,11):
        for exercise in args.exercises:
            print("Loading exercise " + str(exercise) + " of subject " + str(subject))
            # Load the .mat file
            data[exercise] = sio.loadmat(f'./NinaproDB5/s{subject}/S{subject}_E' + str(exercise) + '_A1.mat')

            # Print the keys of the loaded data
            print("Keys of exercise "+ str(exercise) + " " + str(data[exercise].keys()))

            # Access specific variables from the loaded data
            emg_microvolts[exercise] = data[exercise]['emg']
            emg_class[exercise] = data[exercise]['restimulus']


    # %%
    for exercise in args.exercises:
        print("Exercise " + str(exercise))
        print("Shape of emg_microvolts:", emg_microvolts[exercise].shape)
        print("Shape of emg_class:", emg_class[exercise].shape)
        print("Number of gestures:", max(emg_class[exercise])[0])
        
    print("Remember that the rest gesture includes another gesture")

    # %%
    # mat to tensor
    wLen = 250 # Hz
    def getEMG (subject: int, exercise: int):
        sub = str(subject+1)
        mat_data = sio.loadmat('./NinaproDB5/s' + sub + '/S' + sub + '_E' + str(exercise) + '_A1.mat')
        mat_array = mat_data['emg']
        return mat_array

    def getRestimulus (subject: int, exercise: int):
        sub = str(subject+1)
        mat_data = sio.loadmat('./NinaproDB5/s' + sub + '/S' + sub + '_E' + str(exercise) + '_A1.mat')
        mat_array = mat_data['restimulus']
        return mat_array

    # %%
    for i in tqdm(range(0, 10), desc='subject'):
        foldername = 'DatasetsProcessed_hdf5/NinaproDB5/s' + str(i+1) + '/'
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        for j in tqdm(args.exercises, desc='exercise'):
            restimulus_file_path = foldername + 'restimulusS' + str(i+1) + '_E' + str(j) + '.hdf5'
            emg_file_path = foldername + '/emgS' + str(i+1) + '_E' + str(j) + '.hdf5'
            
            restimulus_data = getRestimulus(i, j)
            emg_data = getEMG(i, j)
            
            restimulus_df = pd.DataFrame(restimulus_data)
            emg_df = pd.DataFrame(emg_data)
            
            # restimulus_df.to_csv(restimulus_file_path, index=False)
            # emg_df.to_csv(emg_file_path, index=False)
            # save as hdf5 files
            restimulus_df.to_hdf(restimulus_file_path, key='df', mode='w')
            emg_df.to_hdf(emg_file_path, key='df', mode='w')
        # restimulus_file_path = foldername + 'restimulusS' + str(i+1) + '_E2.hdf5'
        # emg_file_path = foldername + '/emgS' + str(i+1) + '_E2.hdf5'
        
        # restimulus_data = getRestimulus(i)
        # emg_data = getEMG(i)
        
        # restimulus_df = pd.DataFrame(restimulus_data)
        # emg_df = pd.DataFrame(emg_data)
        
        # # restimulus_df.to_csv(restimulus_file_path, index=False)
        # # emg_df.to_csv(emg_file_path, index=False)
        # # save as hdf5 files
        # restimulus_df.to_hdf(restimulus_file_path, key='df', mode='w')
        # emg_df.to_hdf(emg_file_path, key='df', mode='w')


if __name__ == "__main__":
    get()