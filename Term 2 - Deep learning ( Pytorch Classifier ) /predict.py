#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
from torch.autograd import Variable
from torch.optim import lr_scheduler
import time
import json
import argparse
import os

"""Execute shell command"""
"""python predict.py --gpu --category_names cat_to_name.json --image_path 'flowers/test/28/image_05230.jpg'"""

def get_command_line_args():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--top_k', type=int, default=5,
                        help='Top k probabilities')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Category names of flowers')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='run on GPU')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='directory of saved model')
    parser.add_argument('--image_path', type=str,
                        help='image Path')
    return parser.parse_args()


def load_checkpoint(args):
    checkpoint = torch.load(args.save_dir)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']      
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['mapping']   
    return model



#Return Numpy array from image
def process_image(image):
    aspect = image.size[0]/image.size[1]
    if aspect > 0:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
    left_margin = (image.width-224)/2
    top_margin = (image.height-224)/2
    image = image.crop((left_margin, top_margin, left_margin+224, top_margin+224))
    
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    
    image = image.transpose((2, 0, 1))
    
    return image


def predict( model, args, cat_to_name):
    image = Image.open(args.image_path)
    image = process_image(image)
    model.cpu()
    model.eval()
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    inputs = Variable(image_tensor, requires_grad=False)
    inputs = inputs.unsqueeze(0)
    
    ps = torch.exp(model.forward(inputs))
    
    top_probs, top_labels = ps.topk(args.top_k)
    top_probs, top_labels = top_probs.data.numpy().squeeze(), top_labels.data.numpy().squeeze()
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in top_labels]
    print ('Top predictions')
    [print('Flowers Name : ' + cat_to_name[x]) for x in top_classes]
    print('Top Probabilities : ')
    print (top_probs)
    return top_probs, top_classes


def main():
    print ("START")
    args = get_command_line_args()
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    #Check Device
    if args.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f'Using {device}\n')
    model = load_checkpoint(args)
    probs, classes = predict( model, args, cat_to_name)
    print ("ENDED")

    
    
if __name__ == "__main__":
    main()