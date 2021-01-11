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

"""python train.py --arch "vgg16" --gpu --epochs 4 --learning_rate 0.0006 --dropout_prob 0.6"""

"""python train.py --data_dir flowers --arch "vgg16" --gpu --epochs 3 --learning_rate 0.0005 --dropout_prob 0.5"""

def get_command_line_args():
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    architectures = {'vgg16','vgg11'}
    parser.add_argument('--data_dir', type=str, default="flowers",
                        help="data_directory")
    parser.add_argument('--epochs', type=int, default =3,
                        help='number of epochs')
    parser.add_argument('--hidden_units', type=int, default=1000,
                        help='Hidden units of first layer')
    parser.add_argument('--arch', dest='arch', default='vgg16', action='store',
                        choices=architectures,
                        help='Architecture to use')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--dropout_prob', type=float, default=0.5,
                        help='probability of dropping weights')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='run on GPU')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='directory of saved model')
    return parser.parse_args()

def prepare_data(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(20),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    #  Load the datasets with ImageFolder
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=train_transforms)
    image_datasets['val'] = datasets.ImageFolder(valid_dir, transform=val_transforms)
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=test_transforms)

    
    # Define the dataloaders
    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True)
    dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size =32,shuffle = True)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 32, shuffle = True)
    
    return dataloaders, image_datasets

def load_pre_trained_model(args): 

    #Define which model / Arch
    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif args.arch == "vgg11":
      model = models.vgg11(pretrained=True)   
    for param in model.parameters():
        param.requires_grad = False

    #Create model with 102 outputs (Equals to number of flower categories)
    from collections import OrderedDict
    num_features = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(num_features, args.hidden_units)),
                              ('drop', nn.Dropout(p=args.dropout_prob)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(1000, 500)),
                              ('drop', nn.Dropout(p=args.dropout_prob)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(500, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(
            optimizer, step_size=5, gamma=0.1)

    #Check Device
    if args.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return model,device, criterion, optimizer, exp_lr_scheduler, classifier

def do_deep_learning(args):
    dataloaders, image_datasets = prepare_data(args)
    model,device, criterion, optimizer, exp_lr_scheduler, classifier = load_pre_trained_model(args)
    print ("Model Loaded")
    since = time.time()
    print(f'Training with {device}\n')
    epochs = args.epochs
    steps = 0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train()
                model.to(device)
            else:
                model.eval()

            running_error = 0.0
            running_corrects = 0

            for inputs, targets in dataloaders[phase]:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)
                    _, predictions = torch.max(outputs, 1)
                    error = criterion(outputs, targets)

                    if phase == 'train':
                        error.backward()
                        optimizer.step()

                running_error += error.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == targets.data)

            epoch_error = running_error / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print('{} Error: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_error, epoch_acc))
            print('-' * 20)

    time_training = time.time() - since
    print('Training ended in {:.0f}m {:.0f}s'.format(
        time_training// 60, time_training % 60))
    
    model.class_to_idx = image_datasets['train'].class_to_idx

    # Save the checkpoint 
    torch.save({
        'epoch': args.epochs,
        'arch': args.arch,
        'classifier': classifier,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'mapping': model.class_to_idx },
        args.save_dir)



    print ("Model Saved")
    
def main():
    print ("START")
    args = get_command_line_args()
    do_deep_learning(args)
    print ("ENDED")

    
    
if __name__ == "__main__":
    main()



        




        
