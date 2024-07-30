#Code Adapted from itberrios: https://github.com/itberrios/CV_projects/blob/main/multitask_depth_seg/transformations.py

import numpy as np
import torch
import random
from torchvision import transforms
import torchvision.transforms.functional as TF

from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from PIL import Image, ImageOps

def colormap_cityscapes(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)
    cmap[0,:] = np.array([128, 64,128])
    cmap[1,:] = np.array([244, 35,232])
    cmap[2,:] = np.array([ 70, 70, 70])
    cmap[3,:] = np.array([ 102,102,156])
    cmap[4,:] = np.array([ 190,153,153])
    cmap[5,:] = np.array([ 153,153,153])

    cmap[6,:] = np.array([ 250,170, 30])
    cmap[7,:] = np.array([ 220,220,  0])
    cmap[8,:] = np.array([ 107,142, 35])
    cmap[9,:] = np.array([ 152,251,152])
    cmap[10,:] = np.array([ 70,130,180])

    cmap[11,:] = np.array([ 220, 20, 60])
    cmap[12,:] = np.array([ 255,  0,  0])
    cmap[13,:] = np.array([ 0,  0,142])
    cmap[14,:] = np.array([  0,  0, 70])
    cmap[15,:] = np.array([  0, 60,100])

    cmap[16,:] = np.array([  0, 80,100])
    cmap[17,:] = np.array([  0,  0,230])
    cmap[18,:] = np.array([ 119, 11, 32])
    cmap[19,:] = np.array([ 0,  0,  0])
    
    return cmap


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:

    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        #print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

#Start of new code    
class Normalize(object):
    """ Normalizes RGB image to  0-mean 1-std_dev """ 
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        left, mask = sample
            
        return (TF.normalize(left, self.mean, self.std), 
                mask
        )


class AddColorJitter(object):
    """Convert a color image to grayscale and normalize the color range to [0,1].""" 
    def __init__(self, brightness, contrast, saturation, hue):
        ''' Applies brightness, constrast, saturation, and hue jitter to image ''' 
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        left, mask = sample

        return (self.color_jitter(left), 
                mask
        )


class Rescale(object):
    """ Rescales images with bilinear interpolation and masks with nearest interpolation """

    def __init__(self, h, w):
        self.h, self.w = h, w

    def __call__(self, sample):
        left, mask = sample

        return (TF.resize(left, (self.h, self.w)), 
                TF.resize(mask.unsqueeze(0), (self.h, self.w), transforms.InterpolationMode.NEAREST), 
        )


class RandomCrop(object):
    def __init__(self, h, w, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
        self.h = h
        self.w = w
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        left, mask = sample
        i, j, h, w = transforms.RandomResizedCrop.get_params(left, scale=self.scale, ratio=self.ratio)

        return (TF.resized_crop(left, i, j, h, w, (self.h, self.w)), 
                TF.resized_crop(mask.unsqueeze(0), i, j, h, w, (self.h, self.w), interpolation=TF.InterpolationMode.NEAREST),
        )


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
         
        left, mask = sample

        return (transforms.ToTensor()(left), 
                torch.as_tensor(mask, dtype=torch.int64),
        )
    

class ElasticTransform(object):
    def __init__(self, alpha=25.0, sigma=5.0, prob=0.5):
        self.alpha = [1.0, alpha]
        self.sigma = [1, sigma]
        self.prob = prob

    def __call__(self, sample):
        
        if torch.rand(1) < self.prob:

            left, mask = sample
            _, H, W = mask.shape
            displacement = transforms.ElasticTransform.get_params(self.alpha, self.sigma, [H, W])

            # # TEMP
            # print(TF.elastic_transform(left, displacement).shape)
            # print(TF.elastic_transform(mask.unsqueeze(0), displacement, interpolation=TF.InterpolationMode.NEAREST).shape)
            # print(torch.clip(TF.elastic_transform(depth, displacement), 0, depth.max()).shape)

            return (TF.elastic_transform(left, displacement), 
                    TF.elastic_transform(mask.unsqueeze(0), displacement, interpolation=TF.InterpolationMode.NEAREST), 
            ) 
        
        else:
            return sample


# new transform to rotate the images
class RandomRotate(object):
    def __init__(self, angle):
        if not isinstance(angle, (list, tuple)):
            self.angle = (-abs(angle), abs(angle))
        else:
            self.angle = angle

    def __call__(self, sample):
        left, mask = sample

        angle = transforms.RandomRotation.get_params(self.angle)

        return (TF.rotate(left, angle), 
                TF.rotate(mask.unsqueeze(0), angle)
        )
    
    
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        
        if torch.rand(1) < self.prob:
            left, mask = sample
            return (TF.hflip(left), 
                    TF.hflip(mask)
            )
        else:
            return sample
        

class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if torch.rand(1) < self.prob:
            left, mask = sample
            return (TF.vflip(left), 
                    TF.vflip(mask)
                )
        else:
            return sample
        
#Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target