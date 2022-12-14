import importlib
import torch
import sys
sys.path.append('..')
sys.path.append('.')
from diffusers import DiffusionPipeline
from utils import *
from model.unet import UNet2DModel
from schedule.ddim_schedule import DDIM_schedule

LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_config", "from_config"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "OnnxRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
    },
}

class InpaintingPipeline(DiffusionPipeline):
    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: DDIM_schedule,
    ):
        super().__init__()
        # scheduler = scheduler.set_format("pt")
        self.register_modules(
            unet=unet,
            scheduler=scheduler
        )


    @torch.no_grad()
    def inpaint(
        self,
        init_image: torch.FloatTensor,
        mask_image: torch.FloatTensor,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
    ):

        # preprocess image
        image = preprocess_image(init_image)[None].cuda()
        # preprocess mask
        mask = preprocess_mask(mask_image)[None].cuda()
        init_image = image

        # check sizes
        if not mask.shape == image.shape:
            raise ValueError(f"The mask and init_image should be the same size!")

        self.scheduler.set_timesteps(num_inference_steps)
        noise = torch.randn(image.shape, generator=generator).cuda()
        image = self.scheduler.add_noise(image, noise, self.scheduler.timesteps[0])
        step = self.scheduler.timesteps[0]-self.scheduler.timesteps[1]
        for t in self.progress_bar(self.scheduler.timesteps):

            # predict the noise residual
            noise_pred = self.unet(image, t)["sample"]

            # compute the previous noisy sample x_t -> x_t-1
            image = self.scheduler.step(noise_pred, t, image, eta)["prev_sample"]

            # masking
            init_latents_proper = self.scheduler.add_noise(init_image, noise, t)
            image = (init_latents_proper * (1-mask)) + (image * (mask))
            
            if t != self.scheduler.timesteps[0]:
                image = self.scheduler.add_noise_on_sample(image, noise, t, t+step)
                # predict the noise residual
                noise_pred = self.unet(image, t)["sample"]

                # compute the previous noisy sample x_t -> x_t-1
                image = self.scheduler.step(noise_pred, t, image, eta)["prev_sample"]

                # masking
                init_latents_proper = self.scheduler.add_noise(init_image, noise, t)
                image = (init_latents_proper * (1-mask)) + (image * (mask))
                
                
        image = (init_image * (1-mask)) + (image * (mask))
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        return {"sample": image}
 
    @torch.no_grad()
    def test(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.cuda()

        # set step values
        # self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t)['sample']

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to ?? in paper and should be between [0, 1]
            # do x_t -> x_t-1
            sample = self.scheduler.step(model_output, t, image, eta)
            image = sample['prev_sample']
            
            image1 = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
            self.numpy_to_pil(image1)[0].save(f'/mnt/share/shenfeihong/data/test/ad_diff/img_{t}.png')
            pred_origin = sample['pred_original_sample']
            pred_origin = (pred_origin / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
            self.numpy_to_pil(pred_origin)[0].save(f'/mnt/share/shenfeihong/data/test/ad_diff/{t}.png')

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return {'sample':image}

   
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ):
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipeline_utils.ImagePipelineOutput`] instead of a plain tuple.
        Returns:
            [`~pipeline_utils.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple. When returning a tuple, the first element is a list with the
            generated images.
        """

        # Sample gaussian noise to begin loop
        image = torch.randn(
            (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size),
            generator=generator,
        )
        image = image.cuda()

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t)['sample']

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to ?? in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, eta)['prev_sample']

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return {'sample':image}

    def save_pretrained(self, save_directory: Union[str, os.PathLike], epoch):
        """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~DiffusionPipeline.from_pretrained`]` class method.
        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        self.unet.save_pretrained(save_directory, epoch)
        
    def load_pretrained(self, save_directory: Union[str, os.PathLike], epoch):
        """
        Save all variables of the pipeline that can be saved and loaded as well as the pipelines configuration file to
        a directory. A pipeline variable can be saved and loaded if its class implements both a save and loading
        method. The pipeline can easily be re-loaded using the `[`~DiffusionPipeline.from_pretrained`]` class method.
        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        self.unet.load_pretrained(save_directory, epoch)
if __name__=="__main__":
    unet = UNet2DModel().cuda()
    scheduler = DDIM_schedule()
    pipeLine = InpaintingPipeline(unet, scheduler)
    # pipeLine.save_pretrained('./')
    # pipeLine.load_pretrained('./')
    image = Image.open('/mnt/share/shenfeihong/data/abrasion/in_mouth_pic/001dwx010g/cls/lower/36.jpg')
    mask = Image.open('/mnt/share/shenfeihong/data/abrasion/in_mouth_pic/001dwx010g/cls/lower/36.jpg')
    output = pipeLine.inpaint(image, mask)