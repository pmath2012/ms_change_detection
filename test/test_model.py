import torch
import argparse
import os
import cv2
import numpy as np
from dataset import get_siamese_test_loaders
from torch.nn import functional as F
from tqdm import tqdm
from atrous_networks.unet import ASPPUNet
from baselines.unet import UNetC, SwinUNETRC, UNETRC

def predict(model_name, model, dataloader, target_dir, dataset, device, with_boundary=False):
    model.eval()
    for i_batch, sample in tqdm(enumerate(dataloader)):
        if with_boundary:
            image_1, image_2, mask, boundary = sample['image_1'], sample['image_2'], sample['mask'], sample['boundary']
        else:
            image_1, image_2, mask = sample['image_1'], sample['image_2'], sample['mask']
        image_1 = image_1.to(device)
        image_2 = image_2.to(device)
        mask = mask.to(device)
        if with_boundary:
            boundary = boundary.to(device)
            output_m, output_b = model(image_1, image_2)
        else:
            output_m = F.sigmoid(model(image_1, image_2))
        
        output = output_m.data.cpu().view(256,256).numpy()
        output = np.uint8(output*255)
        filename = f'test_{i_batch}.png'
        cv2.imwrite(os.path.join(target_dir, dataset, model_name, filename), output)

def parse_args():
    parser = argparse.ArgumentParser('Test model')
    parser.add_argument('--model_name', type=str, default='unet', help='Model name')
    parser.add_argument('--model_path', type=str, default='unet.pth', help='Model path')
    parser.add_argument('--dataset', type=str, default='svuh', help='Dataset')
    parser.add_argument('--test_file', type=str, default='test.csv', help='Test file')
    parser.add_argument('--data_directory', type=str, default='/home/prateek/from_kipchoge/ms_project/change_balcrop256/', help='Data directory')
    parser.add_argument('--with_boundary', action='store_true', help='Use boundary')
    parser.add_argument('--target_dir', type=str, default='/home/prateek/ms_change_detection/predictions/', help='Output directory')
    parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU to use')
    return parser.parse_args()

if __name__ == '__main__':
    print('Begin Testing ------->')
    args = parse_args()
    device = args.gpu if torch.cuda.is_available() else 'cpu'
    dataset = args.dataset
    if args.model_name == 'unet':
        model = UNetC(in_channels=2, out_channels=1)
    elif args.model_name == 'unetrc':
        model = UNETRC(in_channels=2, out_channels=1)
    elif args.model_name == 'swinunetrc':
        model = SwinUNETRC(in_channels=2, out_channels=1)
    elif args.model_name == 'asppunet':
        model = ASPPUNet(in_channels=2, n_classes=1)
    else:
        raise ValueError("Unsupported model name")
    
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    print('\tModel loaded successfully -------> ')
    test_dataloader = get_siamese_test_loaders(args.test_file, args.data_directory, with_boundary=args.with_boundary)
    predict(args.model_name, model, test_dataloader, args.target_dir, dataset, device, with_boundary=args.with_boundary)

    print('Done!')