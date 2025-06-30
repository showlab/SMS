import os
import argparse
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from sms import SMS, SMSConfig
from utils import (
    tensor_to_pil, pil_to_tensor, clean_gpu, 
    get_cosine_schedule_with_warmup, resize_image, 
    save_concatenated_images, images2gif
)
from diffusers.optimization import get_scheduler


def prepare_latent(sms_module: SMS, img_path: str, device: str) -> Tuple[torch.Tensor, Image.Image]:
    """Load and encode an image into latent space."""
    img = Image.open(img_path).convert('RGB')
    img_tensor = pil_to_tensor(img).to(device)

    h, w = img_tensor.shape[-2:]
    scale = 512 / min(h, w)
    new_size = (int(h * scale), int(w * scale))
    img_tensor = F.interpolate(img_tensor, new_size, mode="bilinear")

    with torch.no_grad():
        x0 = sms_module.encode_image(img_tensor)
    return x0, img


def train(
    sms_module: SMS, src_x0: torch.Tensor, output_dir: str, img_name: str,
    method: str = "sms", lr: float = 1e-1, num_iters: int = 500, 
    optimizer: str = 'AdamW', lambda_dct: float = 0.0005
) -> Tuple[torch.Tensor, Image.Image]:
    """Optimization."""
    clean_gpu()
    tgt_x0 = src_x0.clone().requires_grad_(True)

    # G optimizer
    if optimizer == 'AdamW':
        optimizer = torch.optim.AdamW([tgt_x0], lr=lr)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD([tgt_x0], lr=lr)
    else:
        raise ValueError("Invalid optimizer type. Please choose either 'AdamW' or 'SGD'.")
    
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 100, int(num_iters * 1.5))
    use_lora = method in ["sms", "vsd"]

    if use_lora: # use lora
        # argsparamer for lora
        args_learning_rate_lora = 1e-03
        args_adam_beta1 = 0.9
        args_adam_beta2 = 0.999
        args_adam_weight_decay = 0.0
        args_adam_epsilon = 1e-08
        args_lr_scheduler = "constant"
        args_lr_warmup_steps = 0
        
        # lora optimizer
        optimizer_lora = torch.optim.AdamW(
            sms_module.unet.parameters(),
            lr=args_learning_rate_lora,
            betas=(args_adam_beta1, args_adam_beta2),
            weight_decay=args_adam_weight_decay,
            eps=args_adam_epsilon,
        )
        lr_scheduler_lora = get_scheduler(
            args_lr_scheduler,
            optimizer=optimizer_lora,
            num_warmup_steps=args_lr_warmup_steps,
            num_training_steps=num_iters,
        )

     # Train!
    pbar = tqdm(range(num_iters))
    images = []
    logging.info(f"Starting optimization | method={method} | lr={lr} | iterations={num_iters}")

    try:
        for i in pbar:
            # 1. update target image
            dic = sms_module(
                    tgt_x0=tgt_x0, src_x0=src_x0, return_dict=True, 
                    method=method, epoch_ratio=i / num_iters, lambda_dct=lambda_dct
                )

            grad = dic['grad'].cpu()
            loss = dic['loss']

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            pbar.set_description(f"Loss: {loss.item()}")

            # 2. update LoRA
            if use_lora:
                dic_lora = sms_module(
                    tgt_x0=tgt_x0, src_x0=src_x0, return_dict=True, 
                    method="diffusion"
                )
                
                dic_lora['loss'].backward()
                optimizer_lora.step()
                lr_scheduler_lora.step()
                optimizer_lora.zero_grad()
            
            if (i + 1) % 50 == 0:
                with torch.no_grad():
                    tgt_img = sms_module.decode_latent(tgt_x0)
                    tgt_img = tensor_to_pil(tgt_img)
                    tgt_img = resize_image(tgt_img, 256)
                    images.append(tgt_img)
 
    except KeyboardInterrupt:
        logging.warning('Optimization interrupted by user.')
    finally:
        if images:
            save_concatenated_images(images, os.path.join(output_dir, "slider_" + img_name + ".png"), ncol=5)
            images2gif(images, os.path.join(output_dir, f"slider_{img_name}.gif"))
    return tgt_x0, images[-1]


def main():
    parser = argparse.ArgumentParser(description='Style Matching Score (SMS)')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--method', type=str, default="sms", help='Optimization method')
    parser.add_argument('--guidance_scale', type=float, default=4.5, help='Guidance scale for SMS')
    parser.add_argument('--num_iters', type=int, default=500, help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate for optimization')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer type')
    parser.add_argument('--lambda_dct', type=float, default=0, help='The weight of the Progressive Spectrum Regularization term')
    # input/output
    parser.add_argument('--img_path', type=str, default="./data/cat.jpg", help='Path to the input image')
    parser.add_argument('--image_prompt', type=str, default="a cat", help='Prompt for the input image')
    parser.add_argument('--style_prompt', type=str, default="ghibli style", help='Prompt for the taraget style')
    parser.add_argument('--output_dir', type=str, default="./output", help='Directory to save results')
    # target style distribution
    parser.add_argument('--sd_path', type=str, default="lzyvegetable/backup-stable-diffusion-v1-5", help='Path to Stable Diffusion model for better style representation.')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to LoRA model')
    # sms/vsd related
    parser.add_argument('--lora_rank', type=int, default=8, help='The dimension of the LoRA update matrices')
    parser.add_argument('--lora_alpha', type=float, default=32, help='The alpha constant of the LoRA update matrices.')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = args.output_dir + "/" +  args.style_prompt.split(",")[0]
    os.makedirs(save_dir, exist_ok=True)

    src_prompt = f"Turn into {args.style_prompt}" if args.method == "sms" else args.image_prompt
    tgt_prompt = f"{args.style_prompt}, {args.image_prompt}"

    sms_config = SMSConfig(
        args.sd_path,
        src_prompt=src_prompt,
        tgt_prompt=tgt_prompt,
        guidance_scale=args.guidance_scale,
        device=args.device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        method=args.method,
        lora_path=args.lora_path,
    )
    sms_module = SMS(sms_config)
    
    torch.cuda.empty_cache()
    image_name = os.path.splitext(os.path.basename(args.img_path))[0]
    src_x0, src_img = prepare_latent(sms_module, args.img_path, device=args.device)

    logging.info(f"Stylizing {image_name} | → : '{args.style_prompt}'")
    tgt_x0, tgt_img = train(
        sms_module, src_x0, save_dir, img_name=image_name,
        method=args.method, lr=args.lr, num_iters=args.num_iters,
        optimizer=args.optimizer, lambda_dct=args.lambda_dct
    )

    save_path = os.path.join(save_dir, image_name + ".png")
    save_concatenated_images([src_img, tgt_img.resize(src_img.size)], save_path)
    logging.info(f"Saved stylized image to {save_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()