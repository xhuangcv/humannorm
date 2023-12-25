#!/bin/bash

mkdir pretrained_models
cd pretrained_models/
git lfs clone https://huggingface.co/xanderhuang/normal-adapted-sd1.5
git lfs clone https://huggingface.co/xanderhuang/depth-adapted-sd1.5
git lfs clone https://huggingface.co/xanderhuang/normal-aligned-sd1.5
git lfs clone https://huggingface.co/xanderhuang/controlnet-normal-sd1.5