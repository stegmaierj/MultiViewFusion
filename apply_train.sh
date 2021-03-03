#! /bin/bash
cd /path/to/main/scripts/

# Semi-supervised model using the quad-view dataset
# /path/to/python apply_train.py --in_channels 4 --feat_channels 64 --num_views 4 --model cycleGAN_semi --tile_size 64 64 64 --data_augmentation False --checkpoint_path /train_checkpoints/ --log_path /logs/ --train_gt_dir /datasets/quad_view/train_unpaired_gt/ --val_gt_dir /datasets/quad_view/val_unpaired_gt/ --train_dir /datasets/quad_view/train/ --val_dir /datasets/quad_view/val/

# Self-supervised model using the quad-view dataset
# /path/to/python apply_train.py --in_channels 4 --feat_channels 64 --num_views 4 --model cycleGAN_self --tile_size 64 64 64 --data_augmentation False --checkpoint_path /train_checkpoints/ --log_path /logs/ --train_dir /datasets/quad_view/train/ --val_dir /datasets/quad_view/val/

# Semi-supervised model using the two-view dataset
# /path/to/python apply_train.py --in_channels 2 --feat_channels 16 --num_views 2 --model cycleGAN_semi --tile_size 16 128 960 --data_augmentation False --checkpoint_path /train_checkpoints/ --log_path /logs/ --train_gt_dir /datasets/two_view/train_unpaired_gt/ --val_gt_dir /datasets/two_view/val_unpaired_gt/ --train_dir /datasets/two_view/train/ --val_dir /datasets/two_view/val/

# Self-supervised model using the two-view dataset
# /path/to/python apply_train.py --in_channels 2 --feat_channels 16 --num_views 2 --model cycleGAN_self --tile_size 16 128 960 --data_augmentation False --checkpoint_path /train_checkpoints/ --log_path /logs/ --train_dir /datasets/two_view/train/ --val_dir /datasets/two_view/val/
