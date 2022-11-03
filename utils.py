import math
import torch
from typing import Callable, List, Optional, Tuple, Union
import os

def autoargs(self: object, locals: object) -> dict:
    ''' Put this line in the __init__ method of a class:
    self.__dict__ = autoargs(self, locals())
    to auto-attributes args to class '''
    kwargs = locals
    kwargs.pop('self')  # remove `self` keywork from the dict
    self.__dict__.update(kwargs)
    return self.__dict__

def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].
    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.
    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)

from PIL import Image
import math

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size, 
        num_inference_steps = config.num_inference_steps,
        generator=torch.manual_seed(config.seed),
    )["sample"]

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    
import inspect
from typing import List, Optional, Union

import numpy as np
import PIL
from tqdm.auto import tqdm
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def preprocess_image(image):
    image = transform(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask):
    mask = transform(mask)
    return mask

def cifar():
    import pickle
    import cv2
    with open('/home/disk/data/cifar-10-python/cifar-10-batches-py/data_batch_1', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    print(dict.keys(), np.array(dict[b'data']).shape)
    imgs = np.array(dict[b'data'])
    for i in range(len(imgs)):
        data = imgs[i]
        img = np.zeros((32,32,3))
        R = data[:1024]
        G = data[1024:2048]
        B = data[2048:]
        img[...,0] = B.reshape(32,32)
        img[...,1] = G.reshape(32,32)
        img[...,2] = R.reshape(32,32)
        img = np.array(img).astype(np.uint8)
        print(img.shape)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    return dict

if __name__=='__main__':
    cifar()