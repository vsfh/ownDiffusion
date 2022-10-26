from io import BytesIO
import requests
import PIL
from PIL import Image
import numpy as np
import os
import uuid
import torch

import cv2
from matplotlib import pyplot as plt
from pipeline.inpaint_pipeline import InpaintingPipeline
import glob
from model.unet import UNet2DModel
from schedule.ddim_schedule import DDIM_schedule

scheduler = DDIM_schedule()
model = UNet2DModel(
    sample_size=64,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
      ),
    ).cuda()
pipe = InpaintingPipeline(
    unet=model,
    scheduler=scheduler
)
pipe.load_pretrained('/mnt/share/shenfeihong/weight/diffusion/ddim-in_mouth-64')
pipe.test()
