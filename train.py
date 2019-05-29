from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet_for_train import RetinaNet
from datagen import VocDataset

from torch.autograd import Variable
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    # net.freeze_bn()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        print('batch_idx',batch_idx)
        inputs = Variable(inputs.cuda())

        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        cls_preds = net(inputs)
        loss = criterion(cls_preds, cls_targets)

        loss.backward()

        optimizer.step()

        train_loss += loss.data.item()
        print('train_loss: %.3f | avg_loss: %.3f' % (loss.data.item(), train_loss/(batch_idx+1)))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        cls_preds = net(inputs)
        loss = criterion(cls_preds, cls_targets)
        test_loss += loss.data.item()
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.data.item(), test_loss / (batch_idx + 1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint-resnet31'):
            os.mkdir('checkpoint-resnet31')
        torch.save(state, 'G:\PycharmProjects\others\ckpt.pth')
        best_loss = test_loss
if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Error: CUDA not found!'
    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('==> the first Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    if True:
        trainset = VocDataset(root=r'D:\drone_image_and_annotation_mixed\train', train=True, transform=transform, input_size=1024)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True, num_workers=3,
                                                  collate_fn=trainset.collate_fn)

        testset = VocDataset(root=r'D:\drone_image_and_annotation_mixed\test', train=True, transform=transform, input_size=1024)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, collate_fn=testset.collate_fn)
    else:
        trainset = VocDataset(root=r'D:\data_set_m\train', train=True, transform=transform, input_size=1024)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=3, shuffle=True, num_workers=3,
                                                  collate_fn=trainset.collate_fn)

        testset = VocDataset(root=r'D:\data_set_m\test', train=True, transform=transform, input_size=1024)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1, collate_fn=testset.collate_fn)

    print('train_scale',len(trainset))
    print('test_scale',len(testset))
    # Model
    net = RetinaNet(num_classes=1)

    the_first_time_to_run_this_program=True
    if the_first_time_to_run_this_program==True:
        net.load_state_dict(torch.load(r'G:\PycharmProjects\others\tar_net_dict.pt'))
    else:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(r'G:\PycharmProjects\others\ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        best_loss = float('inf')
        start_epoch = checkpoint['epoch']
    net.cuda()

    criterion = FocalLoss(1)
    optimizer = optim.SGD(net.parameters(), lr=3e-5, momentum=0.9, weight_decay=1e-4)
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)


