'''
Contains the handler function that will be called by the serverless.
'''

import os
import torch
import concurrent.futures
from diffusers import StableDiffusionXLInpaintPipeline,StableDiffusionXLImg2ImgPipeline

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
import base64
from rp_schemas import INPUT_SCHEMA
from io import BytesIO
from PIL import Image

# -------------------------------- Load Models ------------------------------- #
def load_base():
    base_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "viktfb/sdxl-fashion-model",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
    ).to("cuda")
    base_pipe.enable_xformers_memory_efficient_attention()
    return base_pipe


def load_refiner():
    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
    ).to("cuda")
    refiner_pipe.enable_xformers_memory_efficient_attention()
    return refiner_pipe


with concurrent.futures.ThreadPoolExecutor() as executor:
    future_base = executor.submit(load_base)
    future_refiner = executor.submit(load_refiner)

    base = future_base.result()
    refiner = future_refiner.result()


# ---------------------------------- Helper ---------------------------------- #
def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        image_url = rp_upload.upload_image(job_id, image_path)
        image_urls.append(image_url)
    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    base.scheduler = make_scheduler(job_input['scheduler'], base.scheduler.config)

    msg_old = base64.b64decode(job_input['image_url'])
    init_img_old = Image.open(BytesIO(msg_old)).convert("RGB")

    msg_new = base64.b64decode(job_input['mask_url'])
    init_img_new = Image.open(BytesIO(msg_new)).convert("RGB")
    init_img_new = init_img_new[init_img_new > 0] = 255
    
    image = base(
        prompt=job_input['prompt'],
        negative_prompt=job_input['negative_prompt'],
        height=job_input['height'],
        width=job_input['width'],
        num_inference_steps=job_input['num_inference_steps'],
        guidance_scale=job_input['guidance_scale'],
        output_type="latent",
        image=init_img_old,
        mask_image=init_img_new,
        num_images_per_prompt=job_input['num_images'],
        generator=generator
    ).images

    # Refine the image using refiner with refiner_inference_steps
    output = refiner(
        prompt=job_input['prompt'],
        num_inference_steps=job_input['refiner_inference_steps'],
        strength=job_input['strength'],
        image=image,
        num_images_per_prompt=job_input['num_images'],
        generator=generator
    ).images

    image_urls = _save_and_upload_images(output, job['id'])

    return {"image_url": image_urls[0]} if len(image_urls) == 1 else {"images": image_urls}


runpod.serverless.start({"handler": generate_image})
