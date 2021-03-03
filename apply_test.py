import os
import numpy as np
import torch
from skimage import io
from scipy.ndimage import distance_transform_edt
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from torch import nn

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    
    """
    Main testing routine specific for this project
    :param hparams:
    """

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = network(hparams=hparams)
    # model = model.load_from_checkpoint(hparams.checkpoint_path)
    model.load_state_dict(torch.load(hparams.checkpoint_path)['state_dict'], False)
    model = model.cuda()

    # ------------------------
    # 2 INIT DATA TILER
    # ------------------------
    tiler = Tiler(data_dir=hparams.data_dir, num_views=hparams.num_views, outer_tile_size=hparams.tile_size,
    inner_tile_size=hparams.inner_tile_size, overlap=hparams.overlap)
    
    # ------------------------
    # 3 PROCESS EACH IMAGE
    # ------------------------
    for image_idx in range(tiler.__len__()):    

        # Let the dataset know which image is being processing
        tiler.set_data_idx(image_idx)

        # Fusion image            
        predicted_img = np.zeros((hparams.out_channels,) + tiler.dest_size, dtype=np.float32)

        # Normalization map containing the sum of weights at each position
        norm_map = np.zeros((hparams.out_channels,) + tiler.dest_size, dtype=np.float32)
        
        # Weight map
        weight_map = np.zeros((hparams.out_channels,) + tiler.inner_tile_size, dtype=np.float32)

        # The outer one-pixel-wide "layer" is considered as border
        weight_map[:, 1:-1, 1:-1, 1:-1] = 1

        # Compute the distance from each pixel to the border  
        weight_map = distance_transform_edt(weight_map).astype(np.float32)
        
        # The closer to the center of a tile the higher the weight
        weight_map /= weight_map.max()
        weight_map = np.clip(weight_map, 1e-5, 1)

        for tile_idx in range(tiler.num_tiles):
            
            print('Processing tile {0} / {1} ...'.format(tile_idx + 1, tiler.num_tiles))

            # Get the tile
            sample = tiler.__getitem__(tile_idx)
            data = torch.from_numpy(sample['view'][np.newaxis, ...].astype(np.float32)).cuda()

            # Predict the image
            pred_tile, _ = model(data, None, 'BSB')
            pred_tile = pred_tile.cpu().clone().detach().numpy()
            pred_tile = np.squeeze(pred_tile)[tiler.relative_slicing]
            pred_tile = pred_tile[np.newaxis, ...]
    
            # Get the current slice position
            slicing = tuple(map(slice, (0,) + tuple(tiler.inner_start), (hparams.out_channels,) + tuple(tiler.inner_end)))

            # Add weighted tile to the fusion image
            predicted_img[slicing] = predicted_img[slicing] + np.multiply(pred_tile, weight_map)

            # Add up weights
            norm_map[slicing] = norm_map[slicing] + weight_map


        # Normalize and save the predicted image
        predicted_img = np.divide(predicted_img, norm_map)

        # predicted_img = np.transpose(predicted_img, (1, 2, 3, 0))
        
        # predicted_img = (predicted_img - predicted_img.min()) / predicted_img.max()
        predicted_img = np.clip(predicted_img, 0.0, 1.0)

        if hparams.out_channels > 1:
            predicted_img = predicted_img.astype(np.float16)
        else:
            predicted_img = predicted_img.astype(np.float32)

        io.imsave(hparams.output_path +tiler.tag[image_idx]+'.tif', predicted_img)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # These are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '--data_dir',
        type=str,
        default='/datasets/two_view/',
        help='path to the test data set'
    )

    parent_parser.add_argument(
        '--output_path',
        type=str,
        default='/',
        help='output path for test results'
    )
    
    parent_parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='/test_checkpoints/two_view/semi_supervised/_ckpt_epoch_499.ckpt',
        help='path for checkpoints'
    )
    
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of GPUs to use'
    )

    parent_parser.add_argument(
        '--model',
        type=str,
        default='cycleGAN_semi',
        help='which model to load'
    )  

    parent_parser.add_argument(
        '--tile_size', 
        type=int, 
        default=(16, 128, 960), 
        nargs='+',
        help='size of tile cropped from a padded input view'
    )

    parent_parser.add_argument(
        '--inner_tile_size',
        type=int,
        default=(8, 64, 480),
        nargs='+',
        help='size of center region cropped from a tile'
    )

    parent_parser.add_argument(
        '--overlap',
        type=int,
        default=(4, 32, 240),
        nargs='+',
        help='overlap of adjacent tiles')
    
    parent_args = parent_parser.parse_known_args()[0]
    
    # Load the desired network architecture
    if parent_args.model.lower() == 'cyclegan_self':        
        from models.cycleGAN_self import cycleGAN as network
        from dataloaders.dataloader_self import Multiview_Tiler as Tiler
    elif parent_args.model.lower() == 'cyclegan_semi':
        from models.cycleGAN_semi import cycleGAN as network
        from dataloaders.dataloader_semi import Multiview_Tiler as Tiler           
    else:
        raise ValueError('Model {0} unknown.'.format(parent_args.model))

    # Each LightningModule defines arguments relevant to it
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # Convert lists to tuples
    if type(hyperparams.tile_size) is list:
        hyperparams.tile_size = tuple(hyperparams.tile_size)
    if type(hyperparams.inner_tile_size) is list:
        hyperparams.inner_tile_size = tuple(hyperparams.inner_tile_size)        
    if type(hyperparams.overlap) is list:
        hyperparams.overlap = tuple(hyperparams.overlap)

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
