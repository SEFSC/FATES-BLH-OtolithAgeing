# Environment Setup
This was tested on windows 10 with python 3.9

First install the appropriate pytorch version for your setup.  In my case, I used anaconda to install pytorch:
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

Additional packages to install: opencv, matplotlib, pandas, jupyter, tqdm

For SAM segmentation, follow their github for environment setup and to download model weights at https://github.com/facebookresearch/segment-anything.