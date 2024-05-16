import cv2
import torch


class Normalize(object):
    """Normalize the image in a sample to a given intensity.

    Args:
        output_intensity (tuple or int):
    """

    def __init__(self, output_intensity=1):
        assert isinstance(output_intensity, (int, tuple))
        self.output_intensity = output_intensity

    def __call__(self, sample):
        image, masks = sample['image'], sample['mask']

        img = cv2.normalize(image, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        msk = cv2.normalize(masks, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return {'image': img, 'mask': msk.astype(int)}


class NormalizeSiamese(object):
    """Normalize and transform sample

    Args:
        output_intensity (tuple or int):
    """

    def __init__(self, output_intensity=1):
        assert isinstance(output_intensity, (int, tuple))
        self.output_intensity = output_intensity

    def __call__(self, sample):
        image_1, image_2, masks = sample['image_1'], sample['image_2'], sample['mask']

        img1 = cv2.normalize(image_1, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img2 = cv2.normalize(image_2, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        msk = cv2.normalize(masks, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return {'image_1': img1, 'image_2': img2, 'mask': msk.astype(int)}

class NormalizeBASiamese(object):
    """Normalize and transform sample

    Args:
        output_intensity (tuple or int):
    """

    def __init__(self, output_intensity=1, v_flip=False, h_flip=False, p=0.5):
        assert isinstance(output_intensity, (int, tuple))
        self.output_intensity = output_intensity
        self.p = p
        self.v_flip = v_flip
        self.h_flip = h_flip

    def __call__(self, sample):
        image_1, image_2, masks, boundary = sample['image_1'], sample['image_2'], sample['mask'], sample['boundary']

        img1 = cv2.normalize(image_1, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img2 = cv2.normalize(image_2, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        msk = cv2.normalize(masks, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        boundary = cv2.normalize(boundary, None, 0, self.output_intensity, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        if self.v_flip:
            if torch.rand(1) < self.p:
                img1 = cv2.flip(img1, 0)
                img2 = cv2.flip(img2,0)
                msk = cv2.flip(msk,0)
                boundary = cv2.flip(boundary,0)
        if self.h_flip:
            if torch.rand(1) < self.p:
                img1 = cv2.flip(img1, 1)
                img2 = cv2.flip(img2,1)
                msk = cv2.flip(msk,1)
                boundary = cv2.flip(boundary,1)
                
        return {'image_1': img1, 'image_2': img2, 'mask': msk.astype(int), 'boundary': boundary.astype(int)}
