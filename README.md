<h1 align="center"> Human<span style="color: #036bfc;">N</span><span style="color: #3503fc;">o</span><span style="color: #a200ff;">r</span><span style="color: #e250ff;">m</span>: Learning Normal Diffusion Model for High-quality and Realistic 3D Human Generation </h1>

<h3 align="center"> [Project Page](https://humannorm.github.io/) | [Paper](https://arxiv.org/abs/2310.01406) | [Video](https://www.youtube.com/watch?v=2y-0Kfj5-FI) </h3>

CVPR 2024: Official implementation of HumanNorm, a method for generating high-quality and realistic 3D Humans from prompts.

<p align="center">
[Xin Huang](https://xhuangcv.github.io/)<sup>1*</sup>,
[Ruizhi Shao](https://dsaurus.github.io/saurus/)<sup>2*</sup>,
[Qi Zhang](https://qzhang-cv.github.io/)<sup>1</sup>,
[Hongwen Zhang](https://github.com/StevenLiuWen)<sup>2</sup>,
[Ying Feng](https://scholar.google.com.tw/citations?user=PhkrqioAAAAJ&hl=zh-TW)<sup>1</sup>,
[Yebin Liu](https://liuyebin.com/)<sup>2</sup>,
[Qing Wang](https://teacher.nwpu.edu.cn/qwang.html)<sup>1</sup><br>
<sup>1</sup>Northwestern Polytechnical University, <sup>2</sup>Tsinghua University, <sup>*</sup>Equal Contribution <br>
</p>
<p align="center">
    <img src='https://humannorm.github.io/figs/teaser.png' width="800">
</p>

https://github.com/xhuangcv/humannorm/assets/28997098/892cbbfa-05d3-4481-b7f5-fcae739ac8c9

## Method Overview
<p align="center">
    <img src="https://humannorm.github.io/figs/pipeline.png">
</p>


## Installation

**This part is the same as the original [threestudio](https://github.com/threestudio-project/threestudio). Skip it if you already have installed the environment.**

See [installation.md](docs/installation.md) for additional information, including installation via Docker.

- You must have an NVIDIA graphics card with at least 20GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
- Install `Python >= 3.8`.
- (Optional, Recommended) Create a virtual environment:

```sh
pip3 install virtualenv # if virtualenv is installed, skip it
python3 -m virtualenv venv
. venv/bin/activate

# Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
# For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
python3 -m pip install --upgrade pip
```

- Install `PyTorch >= 1.12`. We have tested on `torch1.12.1+cu113` and `torch2.0.0+cu118`, but other versions should also work fine.

```sh
# torch1.12.1+cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# or torch2.0.0+cu118
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

- (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:

```sh
pip install ninja
```

- Install dependencies:

```sh
pip install -r requirements.txt
```
- (Optional) `tiny-cuda-nn` installation might require downgrading pip to 23.0.1

## Download Finetuned Models
You can download our fine-tuned models on HuggingFace: [Normal-adapted-model](https://huggingface.co/xanderhuang/normal-adapted-sd1.5/tree/main), [Depth-adapted-model](https://huggingface.co/xanderhuang/depth-adapted-sd1.5/tree/main), [Normal-aligned-model](https://huggingface.co/xanderhuang/normal-aligned-sd1.5/tree/main) and [ControlNet](https://huggingface.co/xanderhuang/controlnet-normal-sd1.5/tree/main). We provide the script to download load these models.
```sh
./download_models.sh
```
After downloading, the `pretrained_models/` is structured like:
```
./pretrained_models
├── normal-adapted-sd1.5/
├── depth-adapted-sd1.5/
├── normal-aligned-sd1.5/
└── controlnet-normal-sd1.5/
```

## Download Tets
You can download the predefined Tetrahedra for DMTET by
```sh
sudo apt-get install git-lfs # install git-lfs
cd load/
sudo chmod +x download.sh
./download.sh
```
After downloading, the `load/` is structured like:
```
./load
├── lights/
├── shapes/
└── tets
    ├── ...
    ├── 128_tets.npz
    ├── 256_tets.npz
    ├── 512_tets.npz
    └── ...
```

## Quickstart
The directory `scripts` contains scripts used for <u>full-body</u>, <u>half-body</u>, and <u>head-only</u> human generations. The directory `configs` contains parameter settings for all these generations. 
HumanNorm generates 3D humans in three steps including <u>geometry generation</u>, <u>coarse texture generation</u>, and <u>fine texture generation</u>. You can directly execute these three steps using these scripts. For example,
```sh
./script/run_generation_full_body.sh
```
After generation, you can get the result for each step.

https://github.com/xhuangcv/humannorm/assets/28997098/c728fc44-a205-4349-a259-88f121709318


You can also modify the prompt in `run_generation_full_body.sh` to generate other models. The script looks like this:
```sh
#!/bin/bash
exp_root_dir="./outputs"
test_save_path="./outputs/rgb_cache"
timestamp="_20231223"
tag="curry"
prompt="a DSLR photo of Stephen Curry"

# Stage1: geometry generation
exp_name="stage1-geometry"
python launch.py \
    --config configs/humannorm-geometry-full.yaml \
    --train \
    timestamp=$timestamp \
    tag=$tag \
    name=$exp_name \
    exp_root_dir=$exp_root_dir \
    data.sampling_type="full_body" \
    system.prompt_processor.prompt="$prompt, black background, normal map" \
    system.prompt_processor_add.prompt="$prompt, black background, depth map" \
    system.prompt_processor.human_part_prompt=false \
    system.geometry.shape_init="mesh:./load/shapes/full_body.obj"

# Stage2: coarse texture generation
geometry_convert_from="$exp_root_dir/$exp_name/$tag$timestamp/ckpts/last.ckpt" 
exp_name="stage2-coarse-texture"
root_path="./outputs/$exp_name"
python launch.py \
    --config configs/humannorm-texture-coarse.yaml \
    --train \
    timestamp=$timestamp \
    tag=$tag \
    name=$exp_name \
    exp_root_dir=$exp_root_dir \
    system.geometry_convert_from=$geometry_convert_from \
    data.sampling_type="full_body" \
    data.test_save_path=$test_save_path \
    system.prompt_processor.prompt="$prompt" \
    system.prompt_processor.human_part_prompt=false

# Stage3: fine texture generation
ckpt_name="last.ckpt"
exp_name="stage3-fine-texture"
python launch.py \
    --config configs/humannorm-texture-fine.yaml \
    --train \
    system.geometry_convert_from=$geometry_convert_from \
    data.dataroot=$test_save_path \
    timestamp=$timestamp \
    tag=$tag \
    name=$exp_name \
    exp_root_dir=$exp_root_dir \
    resume="$root_path/$tag$timestamp/ckpts/$ckpt_name" \
    system.prompt_processor.prompt="$prompt" \
    system.prompt_processor.human_part_prompt=false
```

## Todo

- [x] Release the reorganized code.
- [ ] Improve the quality of texture generation.
- [ ] Release the finetuning code.

## Citation
If you find our work useful in your research, please cite:
```
@article{huang2023humannorm,
  title={Humannorm: Learning normal diffusion model for high-quality and realistic 3d human generation},
  author={Huang, Xin and Shao, Ruizhi and Zhang, Qi and Zhang, Hongwen and Feng, Ying and Liu, Yebin and Wang, Qing},
  journal={arXiv preprint arXiv:2310.01406},
  year={2023}
}
```
## Acknowledgments

Our project benefits from the amazing open-source projects:

- [ThreeStudio](https://github.com/threestudio-project/threestudio)
- [Diffusers](https://huggingface.co/docs/diffusers/index)

We are grateful for their contribution.
