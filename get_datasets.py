#!/usr/bin/env python
import argparse
import subprocess
import os


def run(): 
    parser = argparse.ArgumentParser()

    parser.add_argument("--all", action="store_true", help="downloads all datasets")
    parser.add_argument("--CapgMyo_B", action="store_true")
    parser.add_argument("--Hyser", action="store_true")
    parser.add_argument("--FlexWearHD_Dataset", action="store_true")
    parser.add_argument("--MyoArmbandDataset", action="store_true")
    parser.add_argument("--NinaproDB2", action="store_true")
    parser.add_argument("--NinaproDB3", action="store_true")
    parser.add_argument("--NinaproDB5", action="store_true")
    parser.add_argument("--OzdemirEMG", action="store_true")
    parser.add_argument("--UCI", action="store_true")

    args = parser.parse_args()

    if not any(vars(args).values()):
        print("You must specify which datasets you would like to download as an argument.")
        
    if args.CapgMyo_B or args.all:
        subprocess.run(['sh', './get_datasets/get_CapgMyo_B.sh'])
        # TODO: call respecitve utils? 

    if args.Hyser or args.all:
        subprocess.run(['sh', './get_datasets/get_Hyser.sh'])

    if args.FlexWearHD_Dataset or args.all:
        subprocess.run(['sh', './get_datasets/get_FlexWearHD_Dataset.sh'])

    if args.MyoArmbandDataset or args.all:
        subprocess.run(['sh', './get_datasets/get_MyoArmbandDataset.sh'])

    if args.NinaproDB2 or args.all:
        subprocess.run(['sh', './get_datasets/get_NinaproDB2.sh'])

    if args.NinaproDB3 or args.all:
        subprocess.run(['sh', './get_datasets/get_NinaproDB3.sh'])

    if args.NinaproDB5 or args.all:
        subprocess.run(['sh', './get_datasets/get_NinaproDB5.sh'])

    if args.OzdemirEMG or args.all:
        subprocess.run(['sh', './get_datasets/get_OzdemirEMG.sh'])
        subprocess.run(['python', './process_Ozdemir.py'])
        
    if args.UCI or args.all:
        subprocess.run(['sh', './get_datasets/get_UCI.sh'])


if __name__ == "__main__":
    run() 

