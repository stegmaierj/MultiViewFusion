import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl

from argparse import ArgumentParser, Namespace
from collections import OrderedDict

from dataloaders.dataloader_self import Multiview_Dataset
from utils.radam import RAdam

import numpy as np
eps = torch.tensor(np.finfo(float).eps).float()


class Discriminator(nn.Module):
    
    def __init__(self, tile_size=(64, 64, 64), in_channels=1, **kwargs):
        super(Discriminator, self).__init__()
        
        self.tile_size = tile_size
        self.in_channels = in_channels
        self.out_size = tuple([int(p/2**4) for p in tile_size])

        # Define layer instances
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm3d(num_features=64)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm3d(num_features=128)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm3d(num_features=256)
        self.leaky3 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm3d(num_features=512)
        self.leaky4 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv6 = nn.Conv3d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sig6 = nn.Sigmoid()


    def forward(self, img):
        
        out = self.conv1(img)
        out = self.leaky1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.leaky2(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.leaky3(out)
        
        out = self.conv4(out)
        out = self.norm4(out)
        out = self.leaky4(out)
        
        out = self.conv6(out)
        out = self.sig6(out)

        return out


class Generator(nn.Module):
    """
    Implementation of the 3D U-Net architecture.
    """

    def __init__(self, in_channels=4, out_channels=1, feat_channels=64, num_views=4,
        out_activation='sigmoid', norm_method='instance', data_augmentation=True, **kwargs):
        super(Generator, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.num_views = num_views
        self.out_activation = out_activation # relu | sigmoid | tanh | hardtanh | none
        self.norm_method = norm_method # instance | batch | none
        
        if self.norm_method == 'instance':
            self.norm = nn.InstanceNorm3d
        elif self.norm_method == 'batch':
            self.norm = nn.BatchNorm3d
        elif self.norm_method == 'none':
            self.norm = nn.Identity
        else:
            raise ValueError('Unknown normalization method "{0}". Choose from "instance|batch|none".'.format(self.norm_method))


        # The forward generator (point spread functions)
        self.PSF_conv = []
        self.data_augmentation = data_augmentation  # If true, the forward generator has to be re-assigned for a new batch
        self.kernels_initilized = False  # Whether the forward generator has been initialized

        # Define layer instances of the inverse generator (deconvolution network)  
        self.c1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=feat_channels, kernel_size=3, stride=1, padding=1),
            self.norm(num_features=feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=feat_channels, out_channels=feat_channels, kernel_size=3, stride=1, padding=1),
            self.norm(num_features=feat_channels),
            nn.ReLU(inplace=True)
            )
        self.d1 = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))


        self.c2 = nn.Sequential(
            nn.Conv3d(in_channels=feat_channels, out_channels=feat_channels*2, kernel_size=3, stride=1, padding=1),
            self.norm(num_features=feat_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=feat_channels*2, out_channels=feat_channels*2, kernel_size=3, padding=1),
            self.norm(num_features=feat_channels*2),
            nn.ReLU(inplace=True)
            )
        self.d2 = nn.Sequential(nn.MaxPool3d(kernel_size=2, stride=2, padding=0))


        self.c3 = nn.Sequential(
            nn.Conv3d(in_channels=feat_channels*2, out_channels=feat_channels*4, kernel_size=3, stride=1, padding=1),
            self.norm(num_features=feat_channels*4),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=feat_channels*4, out_channels=feat_channels*4, kernel_size=3, stride=1, padding=1),
            self.norm(num_features=feat_channels*4),
            nn.ReLU(inplace=True)
            )
        
        
        self.u1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=feat_channels*4, out_channels=feat_channels*2, kernel_size=4, stride=2, padding=1, output_padding=0),
            self.norm(num_features=feat_channels*4),
            nn.ReLU(inplace=True)
            )
        self.c4 = nn.Sequential(
            nn.Conv3d(in_channels=feat_channels*4, out_channels=feat_channels*2, kernel_size=3, stride=1, padding=1),
            self.norm(num_features=feat_channels*2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=feat_channels*2, out_channels=feat_channels*2, kernel_size=3, stride=1, padding=1),
            self.norm(num_features=feat_channels*2),
            nn.ReLU(inplace=True)
            )
        
        
        self.u2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=feat_channels*2, out_channels=feat_channels, kernel_size=4, stride=2, padding=1, output_padding=0),
            self.norm(num_features=feat_channels*2),
            nn.ReLU(inplace=True)
            )
        self.c5 = nn.Sequential(
            nn.Conv3d(in_channels=feat_channels*2, out_channels=feat_channels, kernel_size=3, stride=1, padding=1),
            self.norm(num_features=feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=feat_channels, out_channels=feat_channels, kernel_size=3, stride=1, padding=1),
            self.norm(num_features=feat_channels),
            nn.ReLU(inplace=True)
            )
        
        
        self.out = nn.Sequential(
            nn.Conv3d(in_channels=feat_channels, out_channels=out_channels, kernel_size=1, stride=1)
            )
       

        if self.out_activation == 'relu':
            self.out_fcn = nn.ReLU()
        elif self.out_activation == 'sigmoid':
            self.out_fcn = nn.Sigmoid()
        elif self.out_activation == 'tanh':
            self.out_fcn = nn.Tanh()
        elif self.out_activation == 'leakyrelu':
            self.out_fcn = nn.LeakyReLU(negative_slope=0.1)
        elif self.out_activation == 'none':
            self.out_fcn = None
        else:
            raise ValueError('Unknown output activation "{0}". Choose from "relu|sigmoid|tanh|hardtanh|none".'.format(self.out_activation))


    def set_kernel_weights(self, PSF):
        '''
        Initialize convolutional kernels with the given PSF
        '''

        for i in range(self.num_views):
            psf_size = tuple(PSF[0, i, ...].size())
            len_padding = tuple([(s - 1) // 2 for s in psf_size])

            if not self.kernels_initilized:
                self.PSF_conv.append(nn.Conv3d(in_channels=self.out_channels, out_channels=self.out_channels, \
                    kernel_size=psf_size, stride=1, padding=len_padding).cuda())

            normalized = (PSF[0, i, ...] / PSF[0, i, ...].sum()).unsqueeze(0).unsqueeze(0)  

            self.PSF_conv[i].weight = nn.Parameter(normalized, requires_grad=False)
            self.PSF_conv[i].bias = nn.Parameter(torch.zeros_like(self.PSF_conv[i].bias), requires_grad=False)     

        self.kernels_initilized = True


    def forward(self, img, PSF, direction):
        
        # Blurry-->sharp-->blurry
        if direction == 'BSB':        

            c1 = self.c1(img)
            d1 = self.d1(c1)
            # print('c1 ', c1.size())
            # print('d1 ', d1.size())

            c2 = self.c2(d1)
            d2 = self.d2(c2)
            # print('c2 ', c2.size())
            # print('d2 ', d2.size())      
                
            c3 = self.c3(d2)
            # print('c3 ', c3.size())               
            
            u1 = self.u1(c3)
            c4 = self.c4(torch.cat((u1, c2), dim=1))
            # print('u1 ', u1.size())
            # print('c4 ', c4.size())        
            
            u2 = self.u2(c4)
            c5 = self.c5(torch.cat((u2, c1), dim=1))
            # print('u2 ', u2.size())
            # print('c5 ', c5.size())                       
            
            if self.out_fcn is None:
                sharp_pred = self.out(c5)
            else:
                sharp_pred = self.out_fcn(self.out(c5))

            if PSF is None:
                blurred_with_PSF = None
            else:
                if self.data_augmentation or not self.kernels_initilized:
                    self.set_kernel_weights(PSF) 

                blurred_with_PSF = []
                for i in range(self.num_views):
                    blurred_with_PSF.append(self.PSF_conv[i](sharp_pred))           
                blurred_with_PSF = torch.cat(blurred_with_PSF, dim=1)  


        # Sharp-->blurry-->sharp
        elif direction == 'SBS':

            if PSF is None:
                blurred_with_PSF = None
            else:
                if self.data_augmentation or not self.kernels_initilized:
                    self.set_kernel_weights(PSF)

                blurred_with_PSF = []
                for i in range(self.num_views):
                    blurred_with_PSF.append(self.PSF_conv[i](img))           
                blurred_with_PSF = torch.cat(blurred_with_PSF, dim=1)        

            c1 = self.c1(blurred_with_PSF)
            d1 = self.d1(c1)
            # print('c1 ', c1.size())
            # print('d1 ', d1.size())

            c2 = self.c2(d1)
            d2 = self.d2(c2)
            # print('c2 ', c2.size())
            # print('d2 ', d2.size())      
                
            c3 = self.c3(d2)
            # print('c3 ', c3.size())               
            
            u1 = self.u1(c3)
            c4 = self.c4(torch.cat((u1, c2), dim=1))
            # print('u1 ', u1.size())
            # print('c4 ', c4.size())        
            
            u2 = self.u2(c4)
            c5 = self.c5(torch.cat((u2, c1), dim=1))
            # print('u2 ', u2.size())
            # print('c5 ', c5.size())                       
            
            if self.out_fcn is None:
                sharp_pred = self.out(c5)
            else:
                sharp_pred = self.out_fcn(self.out(c5))

        return sharp_pred, blurred_with_PSF


class cycleGAN(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams

        # Generator
        self.generator = Generator(in_channels=hparams.in_channels, out_channels=hparams.out_channels,
            feat_channels=hparams.feat_channels, num_views=hparams.num_views, out_activation=hparams.out_activation,
            norm_method=hparams.norm_method, data_augmentation=hparams.data_augmentation)

        # Discriminator with full-size tile
        self.discriminator_full = Discriminator(tile_size=hparams.tile_size, in_channels=hparams.out_channels)

        # Discriminator with half-size tile
        half_tile_size = (dim // 2 for dim in hparams.tile_size)
        self.discriminator_half = Discriminator(tile_size=half_tile_size, in_channels=hparams.out_channels)

        # Cache for generated images
        self.last_views = None
        self.last_BSB_sharp_pred = None
        self.last_SBS_sharp_pred = None
        self.last_gt = None
        self.slicing_half = None

        # Cycle loss function
        if hparams.cycle_loss_function == 'l1':
            self.cycle_loss_func = F.l1_loss
        else:
            self.cycle_loss_func = F.mse_loss


    def forward(self, img, PSF=None, direction='BSB'):
        return self.generator(img, PSF, direction)
    
    
    def compute_cycle_loss(self, img_before_cycle, img_after_cycle):  
        return self.cycle_loss_func(img_before_cycle, img_after_cycle, reduction='mean')


    def compute_LSGAN_loss(self, pred, mask):
        return F.mse_loss(pred, mask, reduction='mean')


    def compute_gradient_loss(self, img):
        """
        3D adaption of numpy.gradient()
        """

        dz = torch.zeros_like(img)
        dy = torch.zeros_like(img)
        dx = torch.zeros_like(img)
        out = [dz, dy, dx]
        N = img.ndimension()
        s_dim = N - 2  # Number of spatial dimensions

        # Initialize slice objects with [:, ..., :]
        slice1 = [slice(None)]*N
        slice2 = [slice(None)]*N
        slice3 = [slice(None)]*N
        slice4 = [slice(None)]*N  

        for i, axis in enumerate(range(-s_dim, 0)):          
            
            # 2nd order difference in the interior
            slice1[axis] = slice(1, -1)
            slice2[axis] = slice(None, -2)
            slice3[axis] = slice(1, -1)
            slice4[axis] = slice(2, None)

            out[i][tuple(slice1)] = (img[tuple(slice4)] - img[tuple(slice2)]) / 2

            # 1st order difference on edges
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0

            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[i][tuple(slice1)] = img[tuple(slice2)] - img[tuple(slice3)]

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2

            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[i][tuple(slice1)] = img[tuple(slice2)] - img[tuple(slice3)]

            # Reset the slice object in this dimension to ":"
            slice1[axis] = slice(None)
            slice2[axis] = slice(None)
            slice3[axis] = slice(None)
            slice4[axis] = slice(None)

            zero_map = (out[i]==0)
            out[i][zero_map] = eps            

        return torch.mean(torch.cat(out, dim=1)**2, dim=(1, 2, 3, 4))


    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.current_epoch==0 and batch_idx==0 and optimizer_idx==0:
            print('Training started...')

        # Get input view, PSF and unmatched ground truth of current batch
        self.last_views, last_PSF, last_slicing = batch['view'], batch['psf'], batch['slicing']
        self.slicing_half = last_slicing[0]

        # Train the generator
        if optimizer_idx == 0:
            
            # Generate images
            self.last_BSB_sharp_pred, self.last_BSB_blurred_with_PSF = self.forward(self.last_views, last_PSF, 'BSB')
                
            # Cycle loss
            cycle_loss = self.compute_cycle_loss(self.last_views, self.last_BSB_blurred_with_PSF)

            # WGAN loss
            # Full size
            D_full = self.discriminator_full(self.last_BSB_sharp_pred.detach())
            validity_full = torch.ones_like(D_full).cuda(self.last_BSB_sharp_pred.device.index)
            LSGAN_loss_full = self.compute_LSGAN_loss(D_full, validity_full)

            # Half size
            D_half = self.discriminator_half(self.last_BSB_sharp_pred.detach()[:, :, self.slicing_half[0][0]:self.slicing_half[1][0],
                self.slicing_half[0][1]:self.slicing_half[1][1], self.slicing_half[0][2]:self.slicing_half[1][2]])
            validity_half = torch.ones_like(D_half).cuda(self.last_BSB_sharp_pred.device.index)
            LSGAN_loss_half = self.compute_LSGAN_loss(D_half, validity_half)

            LSGAN_loss = 0.5 * (LSGAN_loss_full + LSGAN_loss_half)

            # Gradient loss
            grad_loss = self.compute_gradient_loss(self.last_BSB_sharp_pred)

            total_loss = self.hparams.cycle_loss_weight * cycle_loss + LSGAN_loss + grad_loss
            tensorboard_logs = {'total_loss': total_loss, 'cycle_loss': cycle_loss, 'LSGAN_loss_G': LSGAN_loss, 'grad_loss': grad_loss}

            return OrderedDict({'loss': total_loss, 'log': tensorboard_logs})


        # Train discriminator with full-size tile
        if optimizer_idx == 1:

            # Sharp prediction
            D_sharp_pred = self.discriminator_full(self.last_BSB_sharp_pred.detach())
            validity_sharp_pred = torch.zeros_like(D_sharp_pred).cuda(self.last_BSB_sharp_pred.device.index)
            LSGAN_loss = self.compute_LSGAN_loss(D_sharp_pred, validity_sharp_pred)

            tensorboard_logs = {'LSGAN_loss_full': LSGAN_loss}
            
            return OrderedDict({'loss': LSGAN_loss, 'log': tensorboard_logs})


        # Train discriminator with half-size tile
        if optimizer_idx == 2:

            # Sharp prediction
            D_sharp_pred = self.discriminator_half(self.last_BSB_sharp_pred.detach()[:, :, self.slicing_half[0][0]:self.slicing_half[1][0],
                self.slicing_half[0][1]:self.slicing_half[1][1], self.slicing_half[0][2]:self.slicing_half[1][2]])
            validity_sharp_pred = torch.zeros_like(D_sharp_pred).cuda(self.last_BSB_sharp_pred.device.index)
            LSGAN_loss = self.compute_LSGAN_loss(D_sharp_pred, validity_sharp_pred)

            tensorboard_logs = {'LSGAN_loss_half': LSGAN_loss}
            
            return OrderedDict({'loss': LSGAN_loss, 'log': tensorboard_logs})            
        

    def validation_step(self, batch, batch_idx):    
        view, PSF, slicing = batch['view'], batch['psf'], batch['slicing']
        slicing_half = slicing[0]
        slicing_quarter = slicing[1]

        # Generate images
        BSB_sharp_pred, BSB_blurred_with_PSF = self.forward(view, PSF, 'BSB')
            
        # Cycle loss
        cycle_loss = self.compute_cycle_loss(view, BSB_blurred_with_PSF)

        # WGAN loss
        # Full size
        D_full = self.discriminator_full(BSB_sharp_pred)
        validity_full_1 = torch.ones_like(D_full).cuda(BSB_sharp_pred.device.index)
        validity_full_0 = torch.zeros_like(D_full).cuda(BSB_sharp_pred.device.index)
        LSGAN_loss_full_G = self.compute_LSGAN_loss(D_full, validity_full_1)
        LSGAN_loss_full_D = self.compute_LSGAN_loss(D_full, validity_full_0)

        # Half size
        D_half = self.discriminator_half(BSB_sharp_pred[:, :, slicing_half[0][0]:slicing_half[1][0],
            slicing_half[0][1]:slicing_half[1][1], slicing_half[0][2]:slicing_half[1][2]])
        validity_half_1 = torch.ones_like(D_half).cuda(BSB_sharp_pred.device.index)
        validity_half_0 = torch.zeros_like(D_half).cuda(BSB_sharp_pred.device.index)
        LSGAN_loss_half_G = self.compute_LSGAN_loss(D_half, validity_half_1)
        LSGAN_loss_half_D = self.compute_LSGAN_loss(D_half, validity_half_0)

        LSGAN_loss_G = 0.5 * (LSGAN_loss_full_G + LSGAN_loss_half_G)

        # Gradient loss
        grad_loss = self.compute_gradient_loss(BSB_sharp_pred)

        return {'cycle_loss': cycle_loss, 'LSGAN_loss_G': LSGAN_loss_G, 'LSGAN_loss_full_D': LSGAN_loss_full_D, 'LSGAN_loss_half_D': LSGAN_loss_half_D, 'grad_loss': grad_loss} 


    def validation_end(self, outputs):
        avg_cycle_loss = torch.stack([x['cycle_loss'] for x in outputs]).mean()
        avg_LSGAN_loss_G = torch.stack([x['LSGAN_loss_G'] for x in outputs]).mean()
        avg_LSGAN_loss_full_D = torch.stack([x['LSGAN_loss_full_D'] for x in outputs]).mean()
        avg_LSGAN_loss_half_D = torch.stack([x['LSGAN_loss_half_D'] for x in outputs]).mean()
        avg_grad_loss = torch.stack([x['grad_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_cycle_loss': avg_cycle_loss, 'val_LSGAN_loss_G': avg_LSGAN_loss_G, 'val_LSGAN_loss_full_D': avg_LSGAN_loss_full_D, 'val_LSGAN_loss_half_D': avg_LSGAN_loss_half_D, 'val_grad_loss': avg_grad_loss}

        return {'avg_val_loss': avg_cycle_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        opt_g = RAdam(self.generator.parameters(), lr=self.hparams.learning_rate)
        opt_d_full = RAdam(self.discriminator_full.parameters(), lr=self.hparams.learning_rate)
        opt_d_half = RAdam(self.discriminator_half.parameters(), lr=self.hparams.learning_rate)              
        return [opt_g, opt_d_full, opt_d_half], []


    @pl.data_loader
    def train_dataloader(self):
         if self.hparams.train_dir is None:
            return None
         else:
            dataset = Multiview_Dataset(data_dir=self.hparams.train_dir, num_views=self.hparams.num_views, tile_size=self.hparams.tile_size, shuffle=True, data_augmentation=True)
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True)


    @pl.data_loader
    def val_dataloader(self):
        if self.hparams.val_dir is None:
            return None
        else:
            dataset = Multiview_Dataset(data_dir=self.hparams.val_dir, num_views=self.hparams.num_views, tile_size=self.hparams.tile_size, shuffle=False, data_augmentation=False)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)


    def on_epoch_end(self):
        
        slice_idx = self.last_views.size()[2] // 2

        # Log BSB sharp prediction
        BSB_prediction_grid = torchvision.utils.make_grid(torch.cat([self.last_BSB_sharp_pred[:, :, slice_idx, :, :] for i in range(3)], dim=1))
        self.logger.experiment.add_image('BSB_sharp_pred', BSB_prediction_grid, self.current_epoch)

        # Log the input view
        img_grid = torchvision.utils.make_grid(torch.cat([self.last_views[:, 0, slice_idx, :, :].unsqueeze(1) for i in range(3)], dim=1))
        self.logger.experiment.add_image('Raw_input_view', img_grid, self.current_epoch)
        
        
    @staticmethod
    def add_model_specific_args(parent_parser): 
        """
        Parameters you define here will be available to your model through self.hparams
        """

        parser = ArgumentParser(parents=[parent_parser])

        # Network params
        parser.add_argument('--in_channels', default=4, type=int)
        parser.add_argument('--out_channels', default=1, type=int)
        parser.add_argument('--feat_channels', default=64, type=int)
        parser.add_argument('--norm_method', default='instance', type=str)
        parser.add_argument('--out_activation', default='sigmoid', type=str)
        parser.add_argument('--cycle_loss_function', default='l1', type=str)
        parser.add_argument('--cycle_loss_weight', default=10.0, type=float)

        # Data
        parser.add_argument('--train_dir', default='/datasets/quad_view/train/', type=str)
        parser.add_argument('--val_dir', default='/datasets/quad_view/val/', type=str)
        parser.add_argument('--data_augmentation', default=True, type=bool)        
        parser.add_argument('--num_views', default=4, type=int)

        # Training params (opt)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--learning_rate', default=0.0001, type=float)
        
        return parser
