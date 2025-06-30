# imageutil

from typing import List

from PIL import Image

import torch.nn.functional as F

import os
import shutil
import math
from PIL import Image
import torch
import logging
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

# tensor <-> PIL Image
def tensor_to_pil(img):
    if img.ndim == 4:
        img = img[0]
    img = img.cpu().permute(1, 2, 0).detach().numpy()
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    return img


def pil_to_tensor(img, device="cpu"):
    device = torch.device(device)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img[None].transpose(0, 3, 1, 2))
    img = img.to(device)
    return img

# from clip_interrogator import Config, Interrogator

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, -1)

def resize_image(image, target_size):
    w, h = image.size
    l = min(w, h)
    new_w, new_h = int(w * target_size / l), int(h * target_size / l)
    return image.resize((new_w, new_h))

def save_concatenated_images(images: List[Image.Image], save_path: str, ncol: int = None):
    concatenated_img = merge_images(images, ncol)
    concatenated_img.save(save_path)

def stack_images_horizontally(images: List[Image.Image]) -> Image.Image:
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGBA", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im

def stack_images_vertically(images: List[Image.Image]) -> Image.Image:
    widths, heights = zip(*(i.size for i in images))
    max_width = max(widths)
    total_height = sum(heights)
    new_im = Image.new("RGBA", (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
    return new_im

def merge_images(images: List[Image.Image], ncol: int = None) -> Image.Image:
    if ncol is None:
        # Default: stack all images in one horizontal row
        return stack_images_horizontally(images)

    # Arrange images into rows of length ncol
    rows = [images[i:i + ncol] for i in range(0, len(images), ncol)]
    row_images = [stack_images_horizontally(row) for row in rows]
    return stack_images_vertically(row_images)


def images2gif(
    images: List, save_path, optimize=True, duration=None, loop=0, disposal=2
):
    if duration is None:
        duration = int(5000 / len(images))
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=duration,
        loop=loop,
        disposal=disposal,
    )

def resize(img):
    h, w = img.shape[2:]
    l = min(h, w)
    h = int(h * 512 / l)
    w = int(w * 512 / l)
    img_512 = F.interpolate(img, size=(h, w), mode="bilinear")
    return img_512


# sys util
import gc
import torch

def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()