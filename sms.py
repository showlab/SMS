import os
import copy

from dataclasses import dataclass
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDIMScheduler, DiffusionPipeline, UNet2DConditionModel
from safetensors.torch import load_file
from jaxtyping import Float
from peft import LoraConfig

# support methods: SMS, VSD, SDS, DDS, PDS

@dataclass
class SMSConfig:
    sd_pretrained_model_or_path: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 500
    min_step_ratio: float = 0.02
    max_step_ratio: float = 0.98

    src_prompt: str = "a photo of a man"
    tgt_prompt: str = "ghibli style, a photo of a man"
    guidance_scale: float = 100
    device: torch.device = torch.device("cuda")

    lora_rank: int = 8
    lora_alpha: float = 32
    lora_path: str = "lora_ckpt/Pyramid_lora_Ghibli_n3.safetensors"
    method: str = "sms"


class SMS(object):
    def __init__(self, config: SMSConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.pipe = DiffusionPipeline.from_pretrained(
            config.sd_pretrained_model_or_path
        ).to(self.device)

        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        self.scheduler.set_timesteps(config.num_inference_steps)
        self.pipe.scheduler = self.scheduler

        self.unet = self.pipe.unet
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae

        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)

        ## construct text features beforehand.
        self.src_prompt = self.config.src_prompt
        self.tgt_prompt = self.config.tgt_prompt

        self.update_text_features(
            src_prompt=self.src_prompt, tgt_prompt=self.tgt_prompt
        )
        self.null_text_feature = self.encode_text("")
        
        # load target style model
        lora_multiplier = 0.8
        self.unet_style, _ = self.load_lora_weights_civitai(self.pipe, config.lora_path, lora_multiplier, config.device, torch.float16, lora_te=False)
        self.unet_style.requires_grad_(False)

        if config.sd_pretrained_model_or_path != "lzyvegetable/backup-stable-diffusion-v1-5":
            # load real model
            self.unet = UNet2DConditionModel.from_pretrained("lzyvegetable/backup-stable-diffusion-v1-5", subfolder="unet").to(self.device)
            self.unet.requires_grad_(False)
            
        if config.method == "sms" or config.method == "vsd":
            # load fake model
            lora_config = LoraConfig(r=self.config.lora_rank, # The dimension of the LoRA update matrices.
                                lora_alpha=self.config.lora_alpha, # The alpha constant of the LoRA update matrices.
                                target_modules=["to_q", "to_v"],
                                )
            self.unet.add_adapter(lora_config)
            self.unet.train()


    def load_lora_weights_civitai(self, pipeline_src, checkpoint_path, multiplier, device, dtype, lora_te=False):

        pipeline = copy.deepcopy(pipeline_src)

        LORA_PREFIX_UNET = "lora_unet"
        LORA_PREFIX_TEXT_ENCODER = "lora_te"
        # load LoRA weight from .safetensors
        state_dict = load_file(checkpoint_path, device=device)

        updates = defaultdict(dict)
        for key, value in state_dict.items():
            # it is suggested to print out the key, it usually will be something like below
            # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

            layer, elem = key.split('.', 1)
            updates[layer][elem] = value

        # directly update weight in diffusers model
        for layer, elems in updates.items():

            if "text" in layer:
                # seems the text coder also quite important: https://github.com/cloneofsimo/lora/discussions/37
                if lora_te:
                    layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
                    curr_layer = pipeline.text_encoder
                else:
                    # WTF break
                    continue
            else:
                layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
                curr_layer = pipeline.unet

            # find the target layer
            temp_name = layer_infos.pop(0)
            while len(layer_infos) > -1:
                try:
                    curr_layer = curr_layer.__getattr__(temp_name)
                    if len(layer_infos) > 0:
                        temp_name = layer_infos.pop(0)
                    elif len(layer_infos) == 0:
                        break
                except Exception:
                    if len(temp_name) > 0:
                        temp_name += "_" + layer_infos.pop(0)
                    else:
                        temp_name = layer_infos.pop(0)

            # get elements for this layer
            weight_up = elems['lora_up.weight'].to(dtype)
            weight_down = elems['lora_down.weight'].to(dtype)
            alpha = elems['alpha']
            if alpha:
                alpha = alpha.item() / weight_up.shape[1]
            else:
                alpha = 1.0

            # update weight
            if len(weight_up.shape) == 4:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            else:
                curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

        return pipeline.unet, pipeline.text_encoder

    def encode_image(self, img_tensor: Float[torch.Tensor, "B C H W"]):
        x = img_tensor
        x = 2 * x - 1
        x = x.float()
        return self.vae.encode(x).latent_dist.sample() * 0.18215

    def encode_text(self, prompt):
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_encoding = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return text_encoding

    def decode_latent(self, latent):
        x = self.vae.decode(latent / 0.18215).sample
        x = (x / 2 + 0.5).clamp(0, 1)
        return x

    def update_text_features(self, src_prompt=None, tgt_prompt=None):
        if getattr(self, "src_text_feature", None) is None:
            assert src_prompt is not None
            self.src_prompt = src_prompt
            self.src_text_feature = self.encode_text(src_prompt)
        else:
            if src_prompt is not None and src_prompt != self.src_prompt:
                self.src_prompt = src_prompt
                self.src_text_feature = self.encode_text(src_prompt)

        if getattr(self, "tgt_text_feature", None) is None:
            assert tgt_prompt is not None
            self.tgt_prompt = tgt_prompt
            self.tgt_text_feature = self.encode_text(tgt_prompt)
        else:
            if tgt_prompt is not None and tgt_prompt != self.tgt_prompt:
                self.tgt_prompt = tgt_prompt
                self.tgt_text_feature = self.encode_text(tgt_prompt)


    def sms_timestep_sampling(self, batch_size, epoch_ratio):
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        timesteps = reversed(self.scheduler.timesteps)

        min_step = (
            1
            if self.config.min_step_ratio <= 0
            else int(len(timesteps) * self.config.min_step_ratio)
        )

        if self.config.method != "sms" or epoch_ratio is None:
            # If no epoch_ratio is provided, use the default max_step_ratio
            max_step = (
                len(timesteps)
                if self.config.max_step_ratio >= 1
                else int(len(timesteps) * self.config.max_step_ratio)
            )
        else:
            # Dynamically adjust the range of sampling
            # As iteration progresses, reduce the max_step to narrow the sampling range
            self.config.max_step_ratio = 0.5 # ==> max 250 # TODO; 500 avoid injecting too much noise
            scale_factor = 1 - epoch_ratio # (total_iterations - iteration) / total_iterations
            max_step = int(len(timesteps) * (self.config.max_step_ratio * scale_factor))

        max_step = max(max_step, min_step + 1)
        
        idx = torch.randint(
            min_step,
            max_step,
            [batch_size],
            dtype=torch.long,
            device="cpu",
        )
        
        t = timesteps[idx].cpu()
        t_prev = timesteps[idx - 1].cpu()

        return t, t_prev


    def __call__(
        self,
        tgt_x0,
        src_x0,
        tgt_prompt=None,
        src_prompt=None,
        reduction="mean",
        return_dict=False,
        method="sms",
        epoch_ratio=None,
        lambda_dct=None,
    ):
        device = self.device
        scheduler = self.scheduler

        # process text.
        self.update_text_features(src_prompt=src_prompt, tgt_prompt=tgt_prompt)
        tgt_text_embedding, src_text_embedding = (
            self.tgt_text_feature,
            self.src_text_feature,
        )
        uncond_embedding = self.null_text_feature

        batch_size = tgt_x0.shape[0]
        # epoch_ratio = epoch_ratio[0] / epoch_ratio[1]
        t, t_prev = self.sms_timestep_sampling(batch_size, epoch_ratio)

        beta_t = scheduler.betas[t].to(device)
        alpha_t = scheduler.alphas[t].to(device)
        alpha_t_prev = scheduler.alphas[t_prev].to(device)
        alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
        alpha_bar_t_prev = scheduler.alphas_cumprod[t_prev].to(device)
        sigma_t = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * beta_t) ** (0.5)

        noise = torch.randn_like(tgt_x0)
        noise_t_prev = torch.randn_like(tgt_x0)
        
        if method == "sms":
            latents_noisy = scheduler.add_noise(tgt_x0, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)

            with torch.no_grad():
                self.unet.disable_adapters()
                teacher_pred = self.unet_style.forward( # style lora
                    latent_model_input,
                    torch.cat([t] * 2).to(device),
                    encoder_hidden_states=text_embeddings,
                ).sample
                teacher_pred_cond, teacher_pred_uncond = teacher_pred.chunk(2)
                # cfg
                teacher_pred = teacher_pred_uncond + self.config.guidance_scale * (
                    teacher_pred_cond - teacher_pred_uncond
                )

                self.unet.enable_adapters()
                lora_pred = self.unet.forward( # current generater
                    latent_model_input,
                    torch.cat([t] * 2).to(device),
                    encoder_hidden_states=text_embeddings,
                ).sample
                lora_pred_cond, lora_pred_uncond = lora_pred.chunk(2)
                lora_pred = lora_pred_uncond + self.config.guidance_scale * (
                    lora_pred_cond - lora_pred_uncond
                )

            # compute the score gradient
            grad = sigma_t**2 * (teacher_pred - lora_pred)
            grad = torch.nan_to_num(grad)
            
            # --- Progressive Spectrum Regularization --- #
            from dct_util import dct_2d, idct_2d, low_pass, high_pass
           
            h = w = src_x0.shape[-1]
            rate = t / 500 # TODO: mask threshold: actually can make it a hyperparameter...
            l_threshold = int(rate * (h + w - 2))
            # apply low-pass filter
            src_x0_dct = low_pass(dct_2d(src_x0, norm='ortho'), l_threshold)
            tgt_x0_dct = low_pass(dct_2d(tgt_x0, norm='ortho'), l_threshold)
            
            criterion_L1 = torch.nn.L1Loss(reduction=reduction)
            psr_loss = criterion_L1(tgt_x0_dct, src_x0_dct) / batch_size * lambda_dct 

            # --- Semantic-Aware Gradient Refinement --- #
            latents_noisy_ref = scheduler.add_noise(src_x0, noise, t)
            latent_model_input_ref = torch.cat([latents_noisy_ref] * 2, dim=0)

            text_embeddings_ref = torch.cat(
                [src_text_embedding, uncond_embedding], dim=0
            ) # "edit prompt: e.g., Turn into ... style" vs " "

            self.unet.disable_adapters()
            noise_pred_ref = self.unet.forward(
                latent_model_input_ref,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=text_embeddings_ref,
            ).sample
                
            noise_pred_text_ref, noise_pred_uncond_ref = noise_pred_ref.chunk(2)
            ref = (noise_pred_text_ref - noise_pred_uncond_ref).detach().squeeze(0).mean(0).abs()
        
            # normalize rel map to [0, 1]
            ref_np = ref.cpu().numpy()
            min_val, max_val = ref_np.min(), ref_np.max()
            norm_ref = (ref_np - min_val) / (max_val - min_val) if max_val > min_val else np.ones_like(ref_np)

            # broadcast
            weight_tensor = torch.from_numpy(norm_ref).float().to(device)       # [64, 64]
            weight_tensor = weight_tensor[None, None, :, :].expand_as(grad)     # [1, 4, 64, 64]

            # element-wise refinement
            weighted_grad = grad * weight_tensor 

            # refined vsd loss
            target = (tgt_x0 - weighted_grad).detach()
            loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size
            # print(f"vsd loss: {loss}, dct loss: {fft_L1_loss}")
            # add the dct loss
            loss += psr_loss
            
            if return_dict:
                dic = {"loss": loss, "grad": grad, "t": t}
                return dic
            else:
                return loss

        elif method == "vsd":
            latents_noisy = scheduler.add_noise(tgt_x0, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)

            with torch.no_grad():
                self.unet.disable_adapters()
                teacher_pred = self.unet_style.forward( # style lora
                    latent_model_input,
                    torch.cat([t] * 2).to(device),
                    encoder_hidden_states=text_embeddings,
                ).sample
                teacher_pred_cond, teacher_pred_uncond = teacher_pred.chunk(2)
                # cfg
                teacher_pred = teacher_pred_uncond + self.config.guidance_scale * (
                    teacher_pred_cond - teacher_pred_uncond
                )

                self.unet.enable_adapters()
                lora_pred = self.unet.forward( # current generater
                    latent_model_input,
                    torch.cat([t] * 2).to(device),
                    encoder_hidden_states=text_embeddings,
                ).sample
                lora_pred_cond, lora_pred_uncond = lora_pred.chunk(2)
                lora_pred = lora_pred_uncond + self.config.guidance_scale * (
                    lora_pred_cond - lora_pred_uncond
                )

            # compute the score gradient
            grad = sigma_t**2 * (teacher_pred - lora_pred)
            grad = torch.nan_to_num(grad)

            # vsd loss
            target = (tgt_x0 - grad).detach()
            loss = 0.5 * F.mse_loss(tgt_x0, target, reduction=reduction) / batch_size

            if return_dict:
                dic = {"loss": loss, "grad": grad, "t": t}
                return dic
            else:
                return loss

        elif method == "diffusion": 
            # forward 
            latents_noisy = scheduler.add_noise(tgt_x0.detach(), noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            text_embeddings = torch.cat([tgt_text_embedding, uncond_embedding], dim=0)

            self.unet.enable_adapters()
            lora_pred = self.unet.forward( # activate adapter
                latent_model_input,
                torch.cat([t] * 2).to(device),
                encoder_hidden_states=text_embeddings,
            ).sample
            lora_pred_cond, lora_pred_uncond = lora_pred.chunk(2)

            lora_pred = alpha_bar_t  * lora_pred_cond
            target = alpha_bar_t * noise

            # compute the loss
            loss = F.mse_loss(lora_pred, target, reduction=reduction) / batch_size
            
            if return_dict:
                dic = {"loss": loss, "t": t}
                return dic
            else:
                return loss

        elif method == "pds":
            # TODO
            pass

        elif method == "dds":
            # TODO
            pass

        elif method == "sds":
            # TODO
            pass

        else:
            raise ValueError(f"Unknown method: {method}. Supported methods: sms, vsd, pds, dds, sds.")
