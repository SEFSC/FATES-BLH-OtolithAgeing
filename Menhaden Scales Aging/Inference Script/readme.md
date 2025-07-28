# Inference Script
Currently has 2 scripts, one for each of chapter one and chapter two.  

Scale_Raw_Image_Preprocessing.py is the script for preprocessing raw scale images in order to crop and pad around the scale of interest.  There are also options for image normalization after cropping.

Scale_Aging_Inference_Script_Image_Only.py is the script for inferencing on the cropped images based on a pretrained model.

All scripts use the same configuration file, configurations.yml.

The scripts can be run with the following console commands:
```
python Scale_Raw_Image_Preprocessing.py --config_path configurations.yml
python Scale_Aging_Inference_Script_Image_Only.py --config_path configurations.yml
```