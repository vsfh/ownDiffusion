from accelerate import Accelerator
import os
import sys
sys.path.append('..')
sys.path.append('.')
import torch
import torch.nn.functional as F
from utils import *
from pipeline.karras_pipeline import KarrasVePipeline
class KarrasVeTrainer():
    def __init__(self) -> None:
        pass
    def inference(self):
        pass
    def train_loop(self, config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps, 
            log_with="tensorboard",
            logging_dir=os.path.join(config.output_dir, "logs")
        )
        if accelerator.is_main_process:
            accelerator.init_trackers("train_example")
        
        # Prepare everything
        # There is no specific order to remember, you just need to unpack the 
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        
        global_step = 0

        # Now you train the model
        for epoch in range(config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")
            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = KarrasVePipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                # if epoch == 0:
                #     pipeline.load_pretrained(config.output_dir, 335)
                    
                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, pipeline)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    pipeline.save_pretrained(config.output_dir, epoch) 

            for step, batch in enumerate(train_dataloader):
                clean_images = batch['images']
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, 1000, (bs,), device=clean_images.device).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                input, input_sigma, label, eff = noise_scheduler.prepare(clean_images, noise, timesteps)
                
                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(input, input_sigma, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, label)
                    accelerator.backward(loss)
                    # print(loss)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

