from diffusers import StableDiffusionPipeline, UNet2DConditionModel

import torch
from diffusers import DPMSolverMultistepScheduler
from PIL import Image
torch.set_grad_enabled(False)

import torchmetrics
from torchmetrics.functional import multimodal
from transformers import AutoProcessor, AutoModel


# a bunch of below is unncessary but was in example of https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/sac%2Blogos%2Bava1-l14-linearMSE.pth
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
from torch.utils.data import Dataset, DataLoader
import json

import clip
import time
import open_clip



from PIL import Image, ImageFile
from PIL import Image
import requests
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

torch.set_grad_enabled(False)
from diffusers import StableDiffusionImg2ImgPipeline

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import skimage
from scipy import ndimage
from scipy.signal import convolve2d

from diffusers import IFPipeline, IFSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil

from diffusers import StableDiffusionXLPipeline


from huggingface_hub import login
with open('hf_auth', 'r') as f:
    auth_token = f.readlines()[0].strip()
login(auth_token)


xgen_sdxl = True

if xgen_sdxl:
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    unet_path = '/export/share/bwallace/gradio_files/dpo_unet/'
    base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                            model_id, torch_dtype=torch.float16,
                            variant="fp16" # , use_safetensors=True
                        ).to("cuda")
else:
    model_id = "runwayml/stable-diffusion-v1-5"
    unet_path = '/export/share/bwallace/gradio_files/ttmpytmpytmpymptmptmpt_epoch_85.ckpt'
    base_pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda')
    base_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(base_pipeline.scheduler.config)
device = "cuda"



print("Loading from", unet_path)
if xgen_sdxl:
    unet = UNet2DConditionModel.from_pretrained(unet_path,
                                                      torch_dtype=torch.float16).to('cuda')
    base_pipeline.unet = unet
else:
    # load model
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    pickscore_processor = AutoProcessor.from_pretrained(processor_name_or_path)
    pickscore_model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
        device
    )
    # Need to disable b/c of all the face skin
    img2img_pipe.old_safety_checker  = img2img_pipe.safety_checker
    img2img_pipe.safety_checker = None

    seg_processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    seg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    face_pixel_thresh = 150


    ckpt = torch.load(unet_path)
    base_pipeline.unet.load_state_dict({k.replace('unet.',''):v for k,v in ckpt['model'].items() if 'unet' in k})
    img2img_pipe.unet.load_state_dict({k.replace('unet.',''):v for k,v in ckpt['model'].items() if 'unet' in k})
    res_64_to_256 = IFSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
    )
    res_64_to_256.enable_model_cpu_offload()
    base_df_pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp16", torch_dtype=torch.float16)
    base_df_pipe.enable_model_cpu_offload()
    # """
    # Below might not even be needed
    safety_modules = {
            "feature_extractor": base_df_pipe.feature_extractor,
            "safety_checker": None, # pipe.safety_checker,
            "watermarker": None, # pipe.watermarker,
    }
    super_res_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
    )
    super_res_pipe.enable_model_cpu_offload()
    super_res_pipe.enable_attention_slicing()
    # super_res_pipe.enable_sequential_cpu_offload()
    super_res_pipe.set_use_memory_efficient_attention_xformers(True)
    # """


sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16")
sdxl_pipe.to("cuda:1")

# Pickscore func
def calc_probs(prompt, images):
    
    # preprocess
    image_inputs = pickscore_processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = pickscore_processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = pickscore_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = pickscore_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores =  (text_embs @ image_embs.T)[0] * pickscore_model.logit_scale.exp() 
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs

# plotting
def image_grid(imgs, rows=2, cols=2,fac=1):
    # fac was trying to increase size but bailed
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(fac*cols * w, fac*rows * h))

    for i, img in enumerate(imgs):
        size = img.size[0]
        # disp_size = fac*size
        grid.paste(img,
                   box=(i % cols * w*fac, i // cols * h*fac))
    return grid

# sharpening
def unsharp_mask(image, amount=1.0, kernel_size=(5, 5), sigma=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        image = np.array(image).astype(np.uint8)
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return Image.fromarray(sharpened)


# Rejection sampling
def generate_and_reject(
    prompt = "Origami depicting Albert Einstein",
        n_batch = 2, # see below
        bs = 8, # total generations = n_batch * bs
        seed_bump = 0, # basically random seed
        correct_face=False, # currently the pipeline detects "face" generically including animal faces so have this as toggle
        display_all=False,
        h=512, w=512,
        n_return_im=1
    ):
    if n_return_im > 1:
        assert not correct_face

    start_time = time.time()
    images = []

    def get_inputs(batch_size=1, seed_set=0):
        generator = [torch.Generator("cuda").manual_seed(i +  (seed_set * 10000)) for i in range(batch_size)]
        prompts = batch_size * [prompt]
        num_inference_steps = 20
        return {"prompt": prompts, "generator": generator, "num_inference_steps": num_inference_steps}


    # Fits on memory but takes 4s instead of 8s
    for batch_i in range(n_batch):
        images.extend(base_pipeline(**get_inputs(batch_size=bs,
                                            seed_set=batch_i+100*seed_bump),
                                    height=h, width=w,).images)
    z = calc_probs(prompt, images).cpu().numpy()
    idx_rank = np.argsort(z)[::-1]
    print("Total time:", time.time() - start_time)

    if display_all:
        display(image_grid(images, n_batch, bs))

    gen_img = images[idx_rank[0]] if (n_return_im==1) else [images[idx_rank[i]] for i in range(n_return_im)]
    if correct_face:
        try:
            gen_img = correct_faces_func(gen_img, face_seed=seed_bump)
        except NotImplementedError:
            print("Too many faces detected, cancelling tuning")
            return gen_img
    return gen_img


# Big tuning faces function
def correct_faces_func(image,
                kernel_size=21,
                min_area=100,
                resize_val = 512,
                num_gens = 8,
                face_seed = 3
                ):
    n_prompt = 1
    seg_inputs = seg_processor(text=['face'],
                               images=[image] ,
                               padding="max_length",
                               return_tensors="pt")
    with torch.no_grad():
      outputs = seg_model(**seg_inputs)
    preds = outputs.logits
    preds = preds.view(1, 1, preds.shape[-2], preds.shape[-1])
    torch_threshold = (torch.sigmoid(preds[0][0]) > (face_pixel_thresh/255))
    clustered = skimage.measure.label(torch_threshold.numpy())

    plt.figure()
    plt.imshow(clustered)
    plt.show()

    def get_min_max_row_col(bool_array):
        active_cols = np.where(bool_array.sum(axis=0))[0]
        min_col, max_col = active_cols[0], active_cols[-1]

        active_rows = np.where(bool_array.sum(axis=1))[0]
        min_row, max_row = active_rows[0], active_rows[-1]

        return  (min_row, max_row), (min_col, max_col)

    kernel = np.ones((kernel_size, kernel_size))

    has_face = False
    for i in range(1, clustered.max()+1):
        cluster_mask = (clustered==i)
        cluster_area = cluster_mask.sum()
        if cluster_area >= min_area:
            if has_face:
                raise NotImplementedError("Error, only supporting 1 face currently")
            has_face = True
            cluster_mask = convolve2d(cluster_mask.astype(int),
                                     kernel.astype(int),
                                     mode='same').astype(bool)
            print(i, cluster_area)
            # print(get_min_max_row_col(cluster_mask))
            (min_row, max_row), (min_col, max_col) = get_min_max_row_col(cluster_mask)

    if not has_face:
        return image
    fac = image.size[-1] / clustered.shape[-1]

    # want to make square
    nonsquare_face_box = [min_col*fac, min_row*fac, max_col*fac, max_row*fac]

    w = (nonsquare_face_box[2] - nonsquare_face_box[0])
    h = (nonsquare_face_box[3] - nonsquare_face_box[1])
    diff = abs(w-h)

    buff1 = diff // 2
    buff2 = diff - buff1
    face_box = nonsquare_face_box.copy()

    if w > h:
        # wider than it is tall
        increase_idx = [1, 3]
    elif w < h:
        # taller than wide
        increase_idx = [0, 2]
    for i,buff in zip(increase_idx, [-buff1, buff2]): # note minus on buff1
        face_box[i] += buff

    face_box = [int(z) for z in face_box]
    face_crop = image.crop(face_box)

    large_face_crop = face_crop.resize((resize_val,resize_val),
                                      Image.Resampling.LANCZOS)
    display(large_face_crop)


    generator = torch.Generator(device=device).manual_seed(face_seed)
    upsampled_face_batch = img2img_pipe(prompt=["Face photograph"]*num_gens, # "supermodel" makes female, same w/ "model", add "realistic"?
                          image=[large_face_crop]*num_gens,

                          strength=0.65, # 0.5 doesn't do much on seed 20, 0.65 does a lot
                         guidance_scale=7.5, # 7.5,
                         generator=generator).images
    upsampled_face_batch = [unsharp_mask(im, 3) for im in upsampled_face_batch]


    blended_batch = []
    for upsampled_face in upsampled_face_batch:
        crop_array = np.array(large_face_crop).astype(float)
        upsampled_array = np.array(upsampled_face).astype(float)

        inputs = seg_processor(text=['face'],
                           images=[upsampled_face],
                           padding="max_length",
                           return_tensors="pt")

        # predict
        with torch.no_grad():
          outputs = seg_model(**inputs)

        preds = outputs.logits

        preds = preds.view(1, 1, preds.shape[-2], preds.shape[-1])


        numpy_mask = (torch.sigmoid(preds[0][0]) > (150/255)).numpy()

        #########

        soft_mask = ndimage.gaussian_filter(numpy_mask.astype(float), 21)
        soft_mask = np.array(Image.fromarray(soft_mask).resize((512, 512)))
        soft_mask = soft_mask[:, :, None].clip(0, 1) # no negative

        # wanted to strengthen original weights
        soft_mask = np.sqrt(soft_mask)  ** 0.5


        #######
        blended_np = soft_mask * upsampled_array + (1- soft_mask) * crop_array


        blended = Image.fromarray(blended_np.astype(np.uint8))
        blended_batch.append(blended)


    pickscore_scores = calc_probs(['Face photograph, realistic']*5, blended_batch)
    pickscore_idx = pickscore_scores.sort(descending=True).indices
    chosen_idx = pickscore_idx[0]
    blended = blended_batch[chosen_idx]
    display(blended)
    paste_mask = Image.fromarray(cluster_mask).resize((resize_val, resize_val))

    # paste_mask = Image.fromarray(clustered==1).resize(face_crop.size)
    canvas_image = image.copy()
    image_to_paste = image.copy()
    # display(image_to_paste)

    image_to_paste.paste(
        blended.resize(face_crop.size, Image.Resampling.LANCZOS),
        face_box
    )
    return image_to_paste


def gen(prompt, resample, upsample, seed, dim, n_return_im=1,
       sdxl=False):
    print(prompt, resample, upsample, seed, dim)
    if xgen_sdxl:
        return base_pipeline(prompt=prompt).images[0]
    elif sdxl:
        assert n_return_im==1
        return sdxl_pipe(prompt=prompt).images[0]
    if n_return_im > 1:
        assert not (resample or upsample)
    im = generate_and_reject(prompt=prompt,
                               seed_bump=seed,
                               correct_face=False,
                               n_batch=4, bs=8,
                               w=dim, h=dim,
                               display_all=False,
                               n_return_im=n_return_im)
    # Adding face upsampling in is TODO
    if resample:
        print("Resample")
        im = torch.tensor(-1 + np.array(im.resize((64, 64), Image.BICUBIC), dtype=np.float) / 127.5).permute(2, 0, 1).unsqueeze(0)
        prompt_embeds, negative_embeds = base_df_pipe.encode_prompt(prompt)
        im = res_64_to_256(
            image=im, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt"
        ).images[0]
        im = pt_to_pil(im.unsqueeze(0))[0]
    if upsample:
        print("Upsample")
        im = super_res_pipe(
            prompt=prompt,
            image=im,
            num_inference_steps=30
        ).images[0]
    return im
