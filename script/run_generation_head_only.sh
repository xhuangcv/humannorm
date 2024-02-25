#!/bin/bash
exp_root_dir="./outputs"
test_save_path="./outputs/rgb_cache"
timestamp="_20231223"
tag="curry"
prompt="a DSLR photo of Stephen Curry"

# Stage1: geometry generation
exp_name="stage1-geometry"
python launch.py \
    --config configs/humannorm-geometry-head.yaml \
    --train \
    timestamp=$timestamp \
    tag=$tag \
    name=$exp_name \
    exp_root_dir=$exp_root_dir \
    data.sampling_type="head_only" \
    system.prompt_processor.prompt="$prompt, black background, normal map" \
    system.prompt_processor_add.prompt="$prompt, black background, depth map" \
    system.geometry.shape_init="mesh:./load/shapes/head_only.obj" \
    system.prompt_processor.human_part_prompt=true

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
    system.geometry.isosurface_resolution=256 \
    data.sampling_type="head_only" \
    data.test_save_path=$test_save_path \
    system.prompt_processor.prompt="$prompt" \
    system.prompt_processor.human_part_prompt=true

# Stage3: fine texture generation
ckpt_name="last.ckpt"
exp_name="stage3-fine-texture"
python launch.py \
    --config configs/humannorm-texture-fine.yaml \
    --train \
    system.geometry_convert_from=$geometry_convert_from \
    system.geometry.isosurface_resolution=256 \
    data.dataroot=$test_save_path \
    timestamp=$timestamp \
    tag=$tag \
    name=$exp_name \
    exp_root_dir=$exp_root_dir \
    resume="$root_path/$tag$timestamp/ckpts/$ckpt_name" \
    system.prompt_processor.prompt="$prompt" \
    system.prompt_processor.human_part_prompt=true
