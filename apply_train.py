import numpy as np
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

SEED = 2020
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.autograd.set_detect_anomaly(True)


def main(hparams):

    """
    Main training routine specific for this project
    :param hparams:
    """

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = network(hparams=hparams)
    
    # Path to the checkpoint for resuming training
    # resume_ckpt = ''

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    
    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.checkpoint_path,
        monitor='val_loss',
        verbose=True,
        save_top_k=-1,
        period=10
    )
    
    logger = TestTubeLogger(
        save_dir=hparams.log_path,
        name = hparams.model.lower()
    )
    
    trainer = Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        use_amp=False,
        max_epochs=hparams.epochs,
        early_stop_callback=False,
        # resume_from_checkpoint=resume_ckpt
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # These are project-wide arguments

    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='/train_checkpoints/',
        help='output path for checkpoints'
    )

    parent_parser.add_argument(
        '--log_path',
        type=str,
        default='/logs/',
        help='output path for logging files'
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
        '--no_resume',
        dest='resume',
        action='store_false',
        default=True,
        help='resume training from a checkpoint'
    )
    
    parent_parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='number of epochs'
    )
    
    parent_parser.add_argument(
        '--model',
        type=str,
        default='cycleGAN_self',
        help='which model to load'
    )

    parent_parser.add_argument(
        '--tile_size', 
        type=int, 
        default=(64, 64, 64), 
        nargs='+',
        help='size of tile sampled from each input view')
    
    
    parent_args = parent_parser.parse_known_args()[0]

    # Load the desired network architecture
    if parent_args.model.lower() == 'cyclegan_self':        
        from models.cycleGAN_self import cycleGAN as network
    elif parent_args.model.lower() == 'cyclegan_semi':
        from models.cycleGAN_semi import cycleGAN as network        
    else:
        raise ValueError('Model {0} unknown.'.format(parent_args.model))
    
    # Each LightningModule defines arguments relevant to it
    parser = network.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)

   
