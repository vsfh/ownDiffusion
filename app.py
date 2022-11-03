import gradio as gr

from io import BytesIO
import requests
import PIL
from PIL import Image
import numpy as np
import os
import uuid
import torch
from torch import autocast
import cv2
from matplotlib import pyplot as plt
from pipeline.inpaint_pipeline import InpaintingPipeline
import glob
from model.unet import UNet2DModel
from schedule.ddim_schedule import DDIM_schedule
from config.ddim_config import DDIM_TrainingConfig

config = DDIM_TrainingConfig()
auth_token = os.environ.get("API_TOKEN") or True
img_size = 256
def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")

device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = DDIM_schedule()
model = UNet2DModel(
    sample_size=img_size,  # the target image resolution
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
epoch = 64
pipe.load_pretrained('/mnt/share/shenfeihong/weight/diffusion/ddim-smile-256', epoch)


def predict(radio, dict):
    if(radio == "draw a mask above"):
        with autocast("cuda"):
            init_image = dict["image"].convert("RGB").resize((img_size, img_size))
            mask = dict["mask"].convert("RGB").resize((img_size, img_size))

    with autocast("cuda"):
        images = pipe.inpaint(init_image=init_image, mask_image=mask, strength=0.8)["sample"]
        # images = pipe(
        #     batch_size = config.eval_batch_size, 
        #     generator=torch.manual_seed(config.seed),
        # )["sample"]
    return images[0]

# examples = [[dict(image="init_image.png", mask="mask_image.png"), "A panda sitting on a bench"]]
css = '''
.container {max-width: 1150px;margin: auto;padding-top: 1.5rem}
#image_upload{min-height:400px}
#image_upload [data-testid="image"], #image_upload [data-testid="image"] > div{min-height: 400px}
#mask_radio .gr-form{background:transparent; border: none}
#word_mask{margin-top: .75em !important}
#word_mask textarea:disabled{opacity: 0.3}
.footer {margin-bottom: 45px;margin-top: 35px;text-align: center;border-bottom: 1px solid #e5e5e5}
.footer>p {font-size: .8rem; display: inline-block; padding: 0 10px;transform: translateY(10px);background: white}
.dark .footer {border-color: #303030}
.dark .footer>p {background: #0b0f19}
.acknowledgments h4{margin: 1.25em 0 .25em 0;font-weight: bold;font-size: 115%}
#image_upload .touch-none{display: flex}
'''
def swap_word_mask(radio_option):
    return gr.update(interactive=False, placeholder="Disabled")

image_blocks = gr.Blocks(css=css)
with image_blocks as demo:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 650px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;">
                  chohotech not for business
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%">
                drawing a mask on bad teeth
              </p>
            </div>
        """
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(source='upload', tool='sketch', elem_id="image_upload", type="pil", label="Upload").style(height=400)
            with gr.Box(elem_id="mask_radio").style(border=False):
                radio = gr.Radio(["draw a mask above"], value="draw a mask above", show_label=False, interactive=True).style(container=False)
            radio.change(None, inputs=[], outputs=image_blocks, _js = """
            () => {
                css_style = document.styleSheets[document.styleSheets.length - 1]
                last_item = css_style.cssRules[css_style.cssRules.length - 1]
                last_item.style.display = ["flex", ""].includes(last_item.style.display) ? "none" : "flex";
            }""")
            btn = gr.Button("Run")
        with gr.Column():
            result = gr.Image(label="Result")
        btn.click(fn=predict, inputs=[radio, image], outputs=result)
    gr.Examples(
        examples=glob.glob("./example/*.jpg"),
        inputs=image)
    gr.HTML(
            """

           """
        )
    
demo.launch()