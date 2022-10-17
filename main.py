import torch

from trainer.inpaint_trainer import InpaintTrainer
from model.unet import UNet2DModel
from schedule.ddim_schedule import DDIM_schedule
from diffusers.optimization import get_cosine_schedule_with_warmup
from data.inpaint_dataset import get_loader
from config.ddim_config import DDIM_TrainingConfig

config = DDIM_TrainingConfig()
dataloader = get_loader()

scheduler = DDIM_schedule()
model = UNet2DModel(sample_size=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(dataloader) * config.num_epochs),
)
trainer = InpaintTrainer()
trainer.train_loop(config, model=model, noise_scheduler=scheduler, optimizer=optimizer, train_dataloader=dataloader, lr_scheduler=lr_scheduler)