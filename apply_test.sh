#! /bin/bash
cd /work/scratch/yang/miccai/

# Semi-supervised model using the quad-view dataset
/work/scratch/yang/miniconda3/envs/pytorch060/bin/python apply_test.py --in_channels 4 --feat_channels 64 --num_views 4 --data_dir /work/scratch/yang/miccai/datasets/quad_view/test/ --checkpoint_path /work/scratch/yang/miccai/test_checkpoints/quad_view/semi_supervised/_ckpt_epoch_89.ckpt --model cycleGAN_semi --tile_size 64 64 64 --inner_tile_size 32 32 32 --overlap 16 16 16

# Self-supervised model using the quad-view dataset
# /work/scratch/yang/miniconda3/envs/pytorch060/bin/python apply_test.py --in_channels 4 --feat_channels 64 --num_views 4 --data_dir /work/scratch/yang/miccai/datasets/quad_view/test/ --checkpoint_path /work/scratch/yang/miccai/test_checkpoints/quad_view/self_supervised/_ckpt_epoch_399.ckpt --model cycleGAN_self --tile_size 64 64 64 --inner_tile_size 32 32 32 --overlap 16 16 16

# Semi-supervised model using the two-view dataset
# /work/scratch/yang/miniconda3/envs/pytorch060/bin/python apply_test.py --in_channels 2 --feat_channels 16 --num_views 2 --data_dir /work/scratch/yang/miccai/datasets/two_view/val/ --checkpoint_path /work/scratch/yang/miccai/test_checkpoints/two_view/semi_supervised/_ckpt_epoch_499.ckpt --model cycleGAN_semi --tile_size 16 128 960 --inner_tile_size 8 64 480 --overlap 4 32 240

# Self-supervised model using the two-view dataset
# /work/scratch/yang/miniconda3/envs/pytorch060/bin/python apply_test.py --in_channels 2 --feat_channels 16 --num_views 2 --data_dir /work/scratch/yang/miccai/datasets/two_view/val/ --checkpoint_path /work/scratch/yang/miccai/test_checkpoints/two_view/self_supervised/_ckpt_epoch_499.ckpt --model cycleGAN_self --tile_size 16 128 960 --inner_tile_size 8 64 480 --overlap 4 32 240
