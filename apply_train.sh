#! /bin/bash
cd /work/scratch/yang/miccai/

# Semi-supervised model using the quad-view dataset
# /work/scratch/yang/miniconda3/envs/pytorch060/bin/python apply_train.py --in_channels 4 --feat_channels 64 --num_views 4 --model cycleGAN_semi --tile_size 64 64 64 --data_augmentation False --checkpoint_path /work/scratch/yang/miccai/train_checkpoints/ --log_path /work/scratch/yang/miccai/logs/ --train_gt_dir /work/scratch/yang/miccai/datasets/quad_view/train_unpaired_gt/ --val_gt_dir /work/scratch/yang/miccai/datasets/quad_view/val_unpaired_gt/ --train_dir /work/scratch/yang/miccai/datasets/quad_view/train/ --val_dir /work/scratch/yang/miccai/datasets/quad_view/val/

# Self-supervised model using the quad-view dataset
# /work/scratch/yang/miniconda3/envs/pytorch060/bin/python apply_train.py --in_channels 4 --feat_channels 64 --num_views 4 --model cycleGAN_self --tile_size 64 64 64 --data_augmentation False --checkpoint_path /work/scratch/yang/miccai/train_checkpoints/ --log_path /work/scratch/yang/miccai/logs/ --train_dir /work/scratch/yang/miccai/datasets/quad_view/train/ --val_dir /work/scratch/yang/miccai/datasets/quad_view/val/

# Semi-supervised model using the two-view dataset
# /work/scratch/yang/miniconda3/envs/pytorch060/bin/python apply_train.py --in_channels 2 --feat_channels 16 --num_views 2 --model cycleGAN_semi --tile_size 16 128 960 --data_augmentation False --checkpoint_path /work/scratch/yang/miccai/train_checkpoints/ --log_path /work/scratch/yang/miccai/logs/ --train_gt_dir /work/scratch/yang/miccai/datasets/two_view/train_unpaired_gt/ --val_gt_dir /work/scratch/yang/miccai/datasets/two_view/val_unpaired_gt/ --train_dir /work/scratch/yang/miccai/datasets/two_view/train/ --val_dir /work/scratch/yang/miccai/datasets/two_view/val/

# Self-supervised model using the two-view dataset
# /work/scratch/yang/miniconda3/envs/pytorch060/bin/python apply_train.py --in_channels 2 --feat_channels 16 --num_views 2 --model cycleGAN_self --tile_size 16 128 960 --data_augmentation False --checkpoint_path /work/scratch/yang/miccai/train_checkpoints/ --log_path /work/scratch/yang/miccai/logs/ --train_dir /work/scratch/yang/miccai/datasets/two_view/train/ --val_dir /work/scratch/yang/miccai/datasets/two_view/val/
