import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from i3d import InceptionI3d

from dataset import Dataset

def run(init_lr=0.1, num_epochs=100, mode='rgb', root='gdrive/MyDrive/sw_capstone', batch_size=1, save_model='', pretrained='imagenet'):
    # setup dataset
    train_transforms = transforms.Compose([RandomCrop(224),
                                           RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([CenterCrop(224)])

    dataset = Dataset(root=root, split='training', mode=mode, transforms=train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


    val_dataset = Dataset(root=root, split='testing', mode=mode, transforms=test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)    

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}
    
    # setup the model
    if mode == 'flow':
        if pretrained == 'imagenet':
            i3d = InceptionI3d(400, in_channels=2)
            i3d.load_state_dict(torch.load(f'{root}/models/flow_imagenet.pt'))
        else:
            i3d = InceptionI3d(157, in_channels=2)
            i3d.load_state_dict(torch.load(f'{root}/models/flow_charades.pt'))
    else:
        if pretrained == 'imagenet':
            i3d = InceptionI3d(400, in_channels=3)
            i3d.load_state_dict(torch.load(f'{root}/models/rgb_imagenet.pt'))
        else:
            i3d = InceptionI3d(157, in_channels=3)
            i3d.load_state_dict(torch.load(f'{root}/models/rgb_charades.pt'))
    i3d.replace_logits(2)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])

    # train it
    best_acc = 0
    train_acc = []
    val_acc = []
    for epoch in range(num_epochs):
        len_data = 0
        print(f'Epoch: {epoch + 1} / {num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        acc = []
        for phase in ['train', 'val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            length_of_data = 0
            num_correct = 0
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                inputs, labels = data
                batch_size = inputs.size(0)
                length_of_data += batch_size
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                t = inputs.size(2)
                labels = Variable(labels.cuda())

                per_frame_logits = i3d(inputs)
                # upsample to input size
                per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data
                # compute classification loss (with max-pooling along time B x C x T)
                cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                tot_cls_loss += cls_loss.data

                loss = 0.5 * loc_loss + 0.5 * cls_loss
                tot_loss += loss.data
                loss.backward()
                if phase == 'train':
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_sched.step()
                pred = torch.max(per_frame_logits, dim=2)[0].detach().argmax(1).cpu().numpy()
                label = torch.max(labels, dim=2)[0].detach().argmax(1).cpu().numpy()
                if pred == label:
                    num_correct += 1
                acc.append(num_correct / float(length_of_data) * 100)

            current_acc = sum(acc) / float(len(acc))
            if phase == 'train':
                print(f'{phase} Accuracy: {current_acc} | Loc Loss: {tot_loc_loss / num_iter:.4f} Cls Loss: {tot_cls_loss / num_iter:.4f} Tot Loss: {(tot_loss) / num_iter:.4f}')              
                train_acc.append(current_acc)
            if phase == 'val':
                print(f'{phase} Accuracy: {current_acc} | Loc Loss: {tot_loc_loss / num_iter:.4f} Cls Loss: {tot_cls_loss / num_iter:.4f} Tot Loss: {(tot_loss) / num_iter:.4f}') 
                if best_acc <= current_acc:
                    best_acc = current_acc
                    torch.save(i3d.module.state_dict(), save_model + str(epoch + 1).zfill(6) + '.pt')
                val_acc.append(current_acc)
            acc.clear()
    return train_acc, val_acc 
