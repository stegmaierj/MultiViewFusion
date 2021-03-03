import numpy as np
import os
import itertools
from skimage import io
from torch.utils.data import Dataset


class Multiview_Dataset(Dataset):
    """
    Sample tiles for network training
    """    
    
    def __init__(self, data_dir, gt_dir, num_views=4, tile_size=(64, 64, 64), shuffle=True, data_augmentation=True):
        
        self.data_dir = data_dir  # Path to raw input views and PSFs
        self.gt_dir = gt_dir  # Path to unmatched ground truth
        self.num_views = num_views  # Number of input views
        self.tile_size = tile_size  # Size of a tile sampled from each input view
        self.shuffle = shuffle  # Flag for shuffling the quadruplets of input views in the training set
        self.data_augmentation = data_augmentation  # Flag for implementing data augmentation

        if self.num_views == 4:  # The embryo dataset is used
            self.img_size = (289,) * 3  # Original size of each input view
            self.dest_size = (256,) * 3  # Actually used size of each input view

        elif self.num_views == 2:  # The nuclei dataset is used
            self.img_size = (140, 140, 1000)  # Original size of each input view
            self.dest_size = (140, 140, 1000)  # Actually used size of each input view
        
        self.data_list = self.load_data()  # List of dictionaries containing paths to views, GTs and PSFs
        
        # Get image statistics from up to first a few view images
        self.data_statistics = {'view': [], 'psf': [], 'gt': []}

        for sample_dict in self.data_list[:3]:
            view = []

            for i in range(self.num_views):
                view.append(io.imread(sample_dict['view'][i]))

            self.data_statistics['view'].append(np.max(view))
            self.data_statistics['gt'].append(np.max(io.imread(sample_dict['gt'])))
        
        # Construct data set statistics
        self.data_statistics['view'] = np.max(self.data_statistics['view'])
        self.data_statistics['gt'] = np.max(self.data_statistics['gt'])
        
        
    def load_data(self):

        file_list = sorted([tiff for tiff in os.listdir(self.data_dir) if tiff.endswith('.tif') and not tiff.startswith('.')])

        psf_name = list(filter(lambda s: 'psf' in s, file_list))  # One set of PSFs for all groups of input views
        view_name = list(filter(lambda s: 'view' in s, file_list))           
        gt_name = sorted([tiff for tiff in os.listdir(self.gt_dir) if 'groundtruth' in tiff and not tiff.startswith('.')])

        data_list = []           

        for i in range(len(gt_name)):

            data_dict = {}
            view_list = []
            psf_list = []

            for j in range(self.num_views):
                view_list.append(self.data_dir + view_name[self.num_views * i + j])
                psf_list.append(self.data_dir + psf_name[j])

            data_dict['view'] = view_list
            data_dict['psf'] = psf_list
            data_dict['gt'] = self.gt_dir + gt_name[i]
            data_list.append(data_dict)
        
        if self.shuffle:
            np.random.shuffle(data_list)        

        return data_list


    def augment_data(self, img):

       # Apply depth flip
        if (self.depth_flip_probability <= 0.5):
            img = np.flip(img, axis=0)

        # Apply vertical flip
        if (self.vertical_flip_probability <= 0.5):
            img = np.flip(img, axis=1)

        # Apply horizontal flip
        if (self.horizontal_flip_probability <= 0.5):
            img = np.flip(img, axis=2)

        return img
    

    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, idx):

        # Indices for slicing an original-sized tile
        crop_start = [int(np.floor((image_dim - dest_dim) / 2)) + np.random.randint(0, np.maximum(1, dest_dim - tile_dim)) for tile_dim, image_dim, dest_dim in zip(self.tile_size, self.img_size, self.dest_size)]
        crop_end = [start + tile_dim for start, tile_dim in zip(crop_start, self.tile_size)]    
        slicing = tuple(map(slice, crop_start, crop_end))            

        # Indices for slicing a half-sized tile sampled from the original-sized one
        crop_start_half = [np.random.randint(0, tile_dim - tile_dim // 2) for tile_dim in self.tile_size]
        crop_end_half = [start_half + tile_dim // 2 for start_half, tile_dim in zip(crop_start_half, self.tile_size)]
        slicing_half = [crop_start_half, crop_end_half]

        # Indices for slicing a quarter-sized tile sampled from the original-sized one (if needed)
        crop_start_quarter = [start_half + np.random.randint(0, tile_dim // 2 - tile_dim // 4) for start_half, tile_dim in zip(crop_start_half, self.tile_size)]
        crop_end_quarter = [start_quarter + tile_dim // 4 for start_quarter, tile_dim in zip(crop_start_quarter, self.tile_size)]
        slicing_quarter = [crop_start_quarter, crop_end_quarter]

        # List of slices for multi-patchGAN
        subslicing = [slicing_half, slicing_quarter]

        # Load a quadruplet of input views and associated PSFs       
        view = []
        psf = []
        for i in range(self.num_views):
            view.append(io.imread(self.data_list[idx]['view'][i]).squeeze()[slicing])
            psf.append(io.imread(self.data_list[idx]['psf'][i]))

        # Load the ground-truth image
        gt = io.imread(self.data_list[idx]['gt'])[slicing]    

        # Apply data augmentation
        if self.data_augmentation:

            # Draw three random variables that determine to enable or disable the respective transforms
            self.vertical_flip_probability = np.random.random_sample()
            self.horizontal_flip_probability = np.random.random_sample()
            self.depth_flip_probability = np.random.random_sample()

            gt = self.augment_data(gt)
            for i in range(self.num_views):
                view[i] = self.augment_data(view[i])
                psf[i] = self.augment_data(psf[i])

        sample = {}    
        sample['view'] = (np.array(view) / self.data_statistics['view']).copy().astype(np.float32)
        sample['psf'] = np.array(psf).copy().astype(np.float32)
        sample['gt'] = (gt[np.newaxis, ...] / self.data_statistics['gt']).copy().astype(np.float32)
        sample['slicing'] = subslicing
            
        return sample








class Multiview_Tiler(Dataset):
    """
    Extracting tiles for volume stitching
    """
    
    def __init__(self, data_dir, num_views=4, outer_tile_size=(64, 64, 64), inner_tile_size=(32, 32, 32), overlap=(16, 16, 16)):
        
        self.data_dir = data_dir  # Path to raw input views and PSFs
        self.num_views = num_views  # Number of input views
        self.outer_tile_size = outer_tile_size  # Size of a tile sampled from each input view
        self.inner_tile_size = inner_tile_size  # Size of the center volume cropped from the network output
        self.overlap = overlap  # Overlapping dimension between two adjacent outer tiles

        if self.num_views == 4:  # The embryo dataset is used
            self.img_size = (289,) * 3  # Original size of each input view
            self.dest_size = (256,) * 3  # Actually used size of each input view

        elif self.num_views == 2:  # The nuclei dataset is used
            self.img_size = (140, 140, 1000)  # Original size of each input view
            self.dest_size = (140, 140, 1000)  # Actually used size of each input view

        # Slice from img_size to dest_size
        crop_start = [int(np.floor((image_dim - dest_dim) / 2)) for image_dim, dest_dim in zip(self.img_size, self.dest_size)]
        crop_end = [start + dest_dim for start, dest_dim in zip(crop_start, self.dest_size)]    
        self.view_slicing = tuple(map(slice, crop_start, crop_end))

        # When the inner tile is close to borders of an input view, the outer tile goes beyond the image volume
        # Thus, the input view in this case has to be padded with zeros
        self.pad_size = [int((outer_dim - inner_dim) / 2) for outer_dim, inner_dim in zip(self.outer_tile_size, self.inner_tile_size)]
        self.pad_tuple = tuple([(pad_width,) * 2 for pad_width in self.pad_size])

        # Final size of input view after padding
        self.extended_size = [dest + pad + pad for dest, pad in zip(self.dest_size, self.pad_size)]

        # Slices for all outer and inner tiles
        self.outer_tile_start, self.outer_tile_end, self.inner_tile_start, self.inner_tile_end = self.get_tile_slicing()
        self.num_tiles = len(self.inner_tile_start)

        # Relative slice of an inner tile within the outer tile
        relative_start = [int((outer_dim - inner_dim) / 2) for outer_dim, inner_dim in zip(outer_tile_size, inner_tile_size)] 
        relative_end = [start + tile_dim for start, tile_dim in zip(relative_start, inner_tile_size)]
        self.relative_slicing = tuple(map(slice, tuple(relative_start), tuple(relative_end)))

        self.data_list = self.load_data()  # List of dictionaries containing paths to views

        # Get image statistics from up to first a few view images
        self.data_statistics = {'view': []}

        for sample_dict in self.data_list[:3]:
            view = []

            for i in range(self.num_views):
                view.append(io.imread(sample_dict['view'][i]))

            self.data_statistics['view'].append(np.max(view))
        
        # Construct data set statistics
        self.data_statistics['view'] = np.max(self.data_statistics['view'])
        
        
    def load_data(self):

        file_list = sorted([tiff for tiff in os.listdir(self.data_dir) if tiff.endswith('.tif') and not tiff.startswith('.')])
        
        view_name = list(filter(lambda s: 'view' in s, file_list))

        self.tag = []  # Tags for input views

        data_list = []
        for i in range(len(view_name) // self.num_views):

            data_dict = {}
            view_list = []

            for j in range(self.num_views):
                view_list.append(self.data_dir + view_name[self.num_views * i + j])

            temp_name = view_name[self.num_views * i]
            self.tag.append(temp_name[:temp_name.find('_')])

            data_dict['view'] = view_list
            data_list.append(data_dict)        

        return data_list


    def get_tile_slicing(self):
        """
        Create a list of coordinates of the upper left corner of all tiles in an image volume
        """

        outer_tile_start = []
        inner_tile_start = []
        for ext_dim, dest_dim, outer, inner, ol in zip(self.extended_size, self.dest_size, self.outer_tile_size, self.inner_tile_size, self.overlap):

            outer_coords = np.arange(1 + (ext_dim - outer) // ol, dtype=np.uint16) * ol
            if (outer + outer_coords[-1]) < ext_dim:
                outer_coords = np.append(outer_coords, ext_dim - outer)
            outer_tile_start.append(outer_coords)

            inner_coords = np.arange(1 + (dest_dim - inner) // (inner - ol), dtype=np.uint16) * (inner - ol)
            if (inner + inner_coords[-1]) < dest_dim:
                inner_coords = np.append(inner_coords, dest_dim - inner)
            inner_tile_start.append(inner_coords)            

        outer_tile_start = list(itertools.product(*outer_tile_start))
        outer_tile_end = []
        for tile_start in outer_tile_start:
            outer_tile_end.append([start + tile_dim for start, tile_dim in zip(tile_start, self.outer_tile_size)])

        inner_tile_start = list(itertools.product(*inner_tile_start))
        inner_tile_end = []
        for tile_start in inner_tile_start:
            inner_tile_end.append([start + tile_dim for start, tile_dim in zip(tile_start, self.inner_tile_size)])

        return outer_tile_start, outer_tile_end, inner_tile_start, inner_tile_end


    def set_data_idx(self, idx):
        self.data_idx = idx


    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, idx):
        
        # Load a quadruplet of input views       
        view_list = []

        # Indices for slicing the center volume of the network output
        self.inner_start = self.inner_tile_start[idx]
        self.inner_end = self.inner_tile_end[idx]

        # Slice for the tile extracted from a padded input view
        outer_slicing = tuple(map(slice, self.outer_tile_start[idx], self.outer_tile_end[idx]))             

        for i in range(self.num_views):
            view = io.imread(self.data_list[self.data_idx]['view'][i]).squeeze()[self.view_slicing]
            view_list.append(np.pad(view, pad_width=self.pad_tuple, mode='constant')[outer_slicing])
  
        sample = {}    
        sample['view'] = np.array(view_list) / self.data_statistics['view']
            
        return sample