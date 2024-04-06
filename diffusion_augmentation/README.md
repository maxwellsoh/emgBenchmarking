# Image Augmentation

1. Run `python unzarr_to_images.py`. Make sure to set the right directories in the command line arguments. Default arguments expect the command to be run from the top level of the repo. 
2. Run `fine-tuning_standard.sh`. 
3. Run `generating_image_data.py`. This may take dozens of hours, depending on number of images. Each image takes 1-2 seconds to generate. 
4. Run plotting scripts if desired.
5. Run `zarr_images.py`.

## Generate Image to Image

3. Run `generating_images_diffusion-from-training.py` and run `generating_images_diffusion-from-validation.py` with changing LOSO-CV subject numbers as needed. 
4. Run  `zarr_images.py` but with `nested_folder=True` enabled. 

    The following is an example of arguments used for nested training folders: `python diffusion_augmentation/zarr_images.py --image_directory=LOSOimages_generated-from-diffusion/OzdemirEMG/cwt/ --output_directory=LOSOimages_zarr_generated-from-diffusion/OzdemirEMG/cwt/ --subject_folder_suffix=_training-img2img --loso_subject_number=13 --guidance_scales=0 --nested_folder=True`. 

    The following is an example of arguments used for the validation folder: `python diffusion_augmentation/zarr_images.py --image_directory=LOSOimages_generated-from-diffusion/OzdemirEMG/cwt/ --output_directory=LOSOimages_zarr_generated-from-diffusion/OzdemirEMG/cwt/ --subject_folder_suffix=_validation-img2img --loso_subject_number=13 --guidance_scales=0`

## Generate Image to Image with ControlNet

2. Run `fine-tuning_subject-to-subject.sh`.