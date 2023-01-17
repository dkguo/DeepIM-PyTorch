#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python modules/object_pose/DeepIM-PyTorch/tools/test_kitchen_images.py --gpu 0 \
  --imgdir $1/ \
  --meta $1/camera_meta.yml \
  --color rgb/*.png \
  --depth depth/*.png \
  --network flownets_rgbd \
  --pretrained modules/object_pose/DeepIM-PyTorch/data/checkpoints/dex_ycb_s0/flownets_dex_ycb_all_rgbd_epoch_20.checkpoint.pth \
  --dataset ycb_object_test \
  --cfg modules/object_pose/DeepIM-PyTorch/experiments/cfgs/dex_ycb_flow_rgbd.yml
