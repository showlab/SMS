#!/bin/sh

DEVICE="cuda:0"

METHOD="sms"
GUIDANCE_SCALE=4.5
NUM_ITERS=500

SD_PATH="ckpt/anything-v4.5"
LORA_PATH="lora_ckpt/Colorwater_v4.safetensors"
LAMBDA_DCT=0.0005   

IMG_PATH="data/dog.jpg"
IMAGE_PROMPT="a close up of a dog with its mouth open."
STYLE_PROMPT="watercolor style"

# 
python stylize.py \
  --sd_path           "$SD_PATH" \
  --device            "$DEVICE" \
  --method            "$METHOD" \
  --guidance_scale    "$GUIDANCE_SCALE" \
  --num_iters         "$NUM_ITERS" \
  --img_path          "$IMG_PATH" \
  --image_prompt      "$IMAGE_PROMPT" \
  --style_prompt      "$STYLE_PROMPT" \
  --lora_path         "$LORA_PATH" \
  --lambda_dct        "$LAMBDA_DCT"
