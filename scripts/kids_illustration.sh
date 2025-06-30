#!/bin/sh

DEVICE="cuda:0"

METHOD="sms"
GUIDANCE_SCALE=4.5
NUM_ITERS=500

SD_PATH="SG161222/Realistic_Vision_V6.0_B1_noVAE"
LORA_PATH="lora_ckpt/COOLKIDS_MERGE_V2.5.safetensors"
LAMBDA_DCT=0.0005   

IMG_PATH="data/bridge.jpg"
IMAGE_PROMPT="a stone bridge over a body of water"
STYLE_PROMPT="kids illustration style, cartoon, simple"

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
