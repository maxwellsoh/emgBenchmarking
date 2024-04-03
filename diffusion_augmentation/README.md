# Image Augmentation

1. Run `python unzarr_to_images.py`. Make sure to set the right directories in the command line arguments. Default arguments expect the command to be run from the top level of the repo. 
2. Run `fine-tuning_standard.sh`. 
3. Run `generating_image_data.py`. This may take dozens of hours, depending on number of images. Each image takes 1-2 seconds to generate. 
4. Run plotting scripts if desired.
5. Run `zarr_images.py`.