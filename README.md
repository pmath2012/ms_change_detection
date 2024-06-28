# Boundary Enhanced Approach for MS(BEAMS) change segmentation
Implementation of BEAMS methods for new lesion segmentation

## TL;DR

To train models without boundary enhancement
```
python train/train_baselines.py 
```
Models implemented - UNet, PFPN, Siamese U-Net, Vision Transformer
To train models with boundary enhancement
```
python train/train_beams.py 
```
Models implemented - PFPN, Siamese U-Net, 

#### Dependencies
The repository is dependent on the glasses repository : https://github.com/FrancescoSaverioZuppichini/glasses/tree/master