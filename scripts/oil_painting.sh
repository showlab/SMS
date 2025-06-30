#!/bin/sh

DEVICE="cuda:0"

METHOD="sms"
GUIDANCE_SCALE=4.5
NUM_ITERS=500

SD_PATH="Lykon/dreamshaper-8"
LORA_PATH="lora_ckpt/fechin.safetensors"
LAMBDA_DCT=0.0005   

IMG_PATH="data/beach.jpg"
IMAGE_PROMPT="the sun is setting over the ocean and waves."
STYLE_PROMPT="oil painting, Fechin, IMPRESSIONISM"

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