from distutils.command.config import config
import torch

from trainer.karras_ve_trainer import KarrasVeTrainer
from trainer.inpaint_trainer import InpaintTrainer
from model.unet import UNet2DModel
from schedule.ddim_schedule import DDIM_schedule
from schedule.karras_ve_schedule import KarrasVeScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from data.inpaint_dataset import get_loader
from config.ddim_config import DDIM_TrainingConfig
from config.sde_config import Karras_TrainingConfig

# config = DDIM_TrainingConfig()
# trainer = InpaintTrainer()
# scheduler = DDIM_schedule()
scheduler = KarrasVeScheduler()
config = Karras_TrainingConfig()
trainer = KarrasVeTrainer()

dataloader = get_loader(config, train=True)


model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
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
    )
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)

trainer.train_loop(config, model=model, noise_scheduler=scheduler, optimizer=optimizer, train_dataloader=dataloader, lr_scheduler=lr_scheduler)