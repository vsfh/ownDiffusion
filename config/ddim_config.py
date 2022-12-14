from dataclasses import dataclass

@dataclass
class DDIM_TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 9
    eval_batch_size = 9  # how many images to sample during evaluation
    num_epochs = 10000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 100
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = '/mnt/share/shenfeihong/weight/diffusion/ddim-smile-256'  # the model namy locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
    data_type = 'smile'
