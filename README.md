# Balanced Image Stylization with Style Matching Score

[Yuxin Jiang](https://yuxinn-j.github.io/), [Liming Jiang](https://liming-jiang.com/), [Shuai Yang](https://williamyang1991.github.io/), [Jia-Wei Liu](https://jia-wei-liu.github.io/), [Ivor W. Tsang](https://www.a-star.edu.sg/cfar/about-cfar/management/prof-ivor-tsang) and [Mike Shou Zheng](https://cde.nus.edu.sg/ece/staff/shou-zheng-mike/)<br>
in ICCV 2025.

[![arXiv](https://img.shields.io/badge/arXiv-2411.17949-b31b1b.svg)](https://arxiv.org/abs/2308.12968)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://yuxinn-j.github.io/projects/SMS.html)

<p align="left">
  <img src="https://github.com/showlab/SMS/blob/main/assets/teaser-f.png" alt="Teaser" width="80%">
</p>

> Style Matching Score (SMS) is a novel optimization method for balanced image stylization with diffusion models. Unlike existing efforts, SMS reframes image stylization as a style distribution matching problem. The target style distribution is estimated from off-the-shelf style-dependent LoRAs via carefully designed score functions. The optimization formulation naturally extends sylization from pixel space to parameter space, making it readily applicable to lighweight generators for efficient one-step stylization, and offering potential for future 3D stylization applications.

-----------------------------------------------------

### Updates

- [ ] Support other score distillation methods (VSD, SDS ...) for completeness.
- [06/2025] Code is released.
- [06/2025] The paper is accepted to ICCV 2025!🎉
- [03/2025] Repo is initialized.

----------------------------------------------------

## 🚀 Get Started
### Environment Setup
```
git clone https://github.com/showlab/SMS.git
cd SMS

conda create -n sms python=3.10 -y
conda activate sms
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps
```
Note: Optimization requires **~17GB** of GPU memory. A 24GB GPU is recommended.
### HuggingFace Cache Configuration (optional)
The program will automatically download pretrained SD models during optimization. If your default disk is low on space, redirect your HuggingFace model cache:
```
export HF_HOME=/your/large/storage/cache
```


## 🎨 Stylization

### 1. Quick Demos
Each script runs optimization for a specific style, saving both the final result and every-50-iteration previews (as `.png` and `.gif`) in the `output/` folder.
We use off-the-shelf style LoRA from [Civitai](https://civitai.com/). Feel free to explore more styles on Civitai or train your own LoRA models to combine with SMS!
#### Pixel Art Style
```
wget "https://civitai.com/api/download/models/31228?type=Model&format=SafeTensor&size=full&fp=fp16" -O ./lora_ckpt/3232pixel.safetensors
bash scripts/pixel_art.sh
```
#### Oil Painting Style
```
wget "https://civitai.com/api/download/models/90795?type=Model&format=SafeTensor" -O ./lora_ckpt/fechin.safetensors
bash scripts/oil_painting.sh
```
#### Watercolor Style
```
wget "https://civitai.com/api/download/models/21173?type=Model&format=SafeTensor&size=full&fp=fp16" -O  ./lora_ckpt/Colorwater_v4.safetensors
bash scripts/watercolor.sh
```
#### Kids Illustration Style
```
wget "https://civitai.com/api/download/models/67980?type=Model&format=SafeTensor" -O ./lora_ckpt/COOLKIDS_MERGE_V2.5.safetensors
bash scripts/kids_illustration.sh
```

### 2. Custom Stylization
The `--lambda_dct` flag controls the strength of Progressive Spectrum Regularization, enabling a flexible **content-style trade-off** based on user preference:

- `0` results in stronger stylization (more abstract and expressive)

- Higher values preserve finer content structure

See `stylize.py` for the full list of arguments and usage examples.

## 🎓 Citation
If you find our work useful in your research, please consider citing our paper:
```
@article{jiang2025balanced,
  title={Balanced Image Stylization with Style Matching Score},
  author={Jiang, Yuxin and Jiang, Liming and Yang, Shuai and Liu, Jia-Wei and Tsang, Ivor and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2503.07601},
  year={2025}
}
```

## 👏 Acknowledgements
Our idea is implemented based on [PDS](https://github.com/KAIST-Visual-AI-Group/PDS). Thanks for their great open-source work!
We also gratefully acknowledge the creators of the open-source style LoRA models and base SD models used in this work.