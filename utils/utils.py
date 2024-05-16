import numpy as np
import cv2
from ..losses import BCE, FBetaLoss, DiceLoss
from monai.losses import DiceFocalLoss, DiceCELoss
from skimage import io


def load_image(img_name):
    if img_name.endswith("png"):
        image = io.imread(img_name)
    elif img_name.endswith("npz"): 
        image = np.load(img_name)['arr_0']
    else:
        raise ValueError("Not yet implemented")
    return image


def get_loss_functions(mask_loss_name, boundary_loss_name):
    if mask_loss_name == 'f0.5':
        mask_loss = FBetaLoss(beta=0.5)
    elif mask_loss_name == 'f1':
        mask_loss = FBetaLoss(beta=1.0)
    elif mask_loss_name == 'f2':
        mask_loss = FBetaLoss(beta=2.0)
    elif mask_loss_name == 'dice':
        mask_loss = DiceLoss()
    elif mask_loss_name == 'DiceFocalLoss':
        mask_loss = DiceFocalLoss(include_background=False, to_onehot_y=False, sigmoid=True)
    elif mask_loss_name == 'DiceCELoss':
        mask_loss = DiceCELoss(include_background=False, to_onehot_y=False, sigmoid=True)
    else:
        raise ValueError("Unsupported mask loss")

    if boundary_loss_name == 'BCE':
        boundary_loss = BCE()
    else:
        raise ValueError("Unsupported boundary loss")

    return mask_loss, boundary_loss