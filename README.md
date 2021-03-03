# MultiViewFusion
Multiview Deconvolution and Fusion using Semi- and Self-Supervised Generative Adversarial Networks

## Environment
Python 3.7.6  
PyTorch 1.3.1  
PyTorch-Lightning 0.6.0  

## Use Instruction
Run the shell scripts apply_train.sh or apply_test.sh to start training or testing, respectively.  
The parser arguments are explained in Python scripts apply_train.py, apply_test.py, models/cycleGAN_semi.py and models/cycleGAN_self.py.  
Note that: the default network hyperparameters in the shell script apply_test.sh should not be modified; otherwise, the models saved in the checkpoints cannot be loaded correctly.  

## Data Naming Scheme
View images: *tag*\_view\_*angle*.tif   
Ground-truth images: *tag*\_groundtruth.tif  
PSF: psf\_*angle*.tif

## Quad-view Dataset
The quad-view embryo dataset is generated using the Java project in multiview_simution.zip provided in Preibisch, S., Amat, F., Stamataki, E., Sarov, M., Singer, R. H., Myers, E., & Tomancak, P. (2014). Efficient Bayesian-based multiview deconvolution. Nature methods, 11(6), 645-648.
