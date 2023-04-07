# LG-DSTF
Label-Guided Dynamic Spatial-Temporal Fusion for Video-Based Facial Expression Recognition

## Requirements
- Python >= 3.6
- PyTorch >= 1.2
- torchvision >= 0.4.0

## Training

- Step 1: download [DFEW](https://dfew-dataset.github.io) dataset.

- Step 2: download spatial pre-trained model from
   [Google Drive](https://drive.google.com/file/d/1I9TBE0DtlsGDxZ8I_xnGKCm1_sNmDsoI/view?usp=sharing), and put it to***./checkpoint***.
    
- Step 3: change ***data_path*** in *train.py* to your path 

- Step 4: run ```python train.py ```


**Acknowledgments**

Our work is based on the following work, thanks for the code:

https://github.com/zengqunzhao/Former-DFER
