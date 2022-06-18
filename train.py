# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""
import wandb
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, twoComponentLoss, get_training_dataloader, get_splitted_dataloaders, get_test_dataloader, get_coarse_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

#from entropy_2_levels import entropy2lvl
#import entropy_2_levels as myEntropy
from models.resnet import ResNet, BasicBlock

superclass = [ 4,  1, 14,  8,  0,  #номер суперкласса соответствует номеру в иерархии на сайте (морские млекопитающие=0, рыбы=1 и т.д.)
               6,  7,  7, 18,  3,  #номер класса соответствует лейблам в датасете
               3, 14,  9, 18,  7, 
              11,  3,  9,  7, 11,  
               6, 11,  5, 10,  7,  
               6, 13, 15,  3, 15,
               0, 11,  1, 10, 12, 
              14, 16,  9, 11,  5,
               5, 19,  8,  8, 15, 
              13, 14, 17, 18, 10,
              16,  4, 17,  4,  2,  
               0, 17,  4, 18, 17,
              10,  3,  2, 12, 12, 
              16, 12,  1,  9, 19,
               2, 10,  0,  1, 16, 
              12,  9, 13, 15, 13,
              16, 19,  2,  4,  6, 
              19,  5,  5,  8, 19,
              18,  1,  2, 15,  6,  
               0, 17,  8, 14, 13]

def train(epoch, trainloader):

    start = time.time()
    net.train()
    for batch_index, (images, superclass_labels, labels) in enumerate(trainloader):

        if args.gpu:
            labels = labels.cuda()
            superclass_labels = superclass_labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function2(outputs, labels)  #set func
        wandb.log({"trainloss": loss})
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(trainloader) + batch_index + 1

        last_layer = list(net.children())[-1]
        
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(trainloader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(testloader, only_coarse, epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    
    correct = 0.0
    if only_coarse==False:
        correctCoarse = 0.0

    for (images, labels) in testloader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        
        if only_coarse==False:
            predsCoarse = [superclass[preds[i]] for i in range(len(labels))]
            realCoarse = [superclass[labels[i]] for i in range(len(labels))]
        
            predsCoarse=torch.tensor(predsCoarse).cuda()
            realCoarse=torch.tensor(realCoarse).cuda()
        
        correct += preds.eq(labels).sum()
        if only_coarse==False:
            correctCoarse += predsCoarse.eq(realCoarse).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    
    if only_coarse==False:
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy100: {:.4f}, Accuracy20: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(testloader.dataset),
            correct.float() / len(testloader.dataset),
            correctCoarse.float() / len(testloader.dataset),
            finish - start
        ))
    else:
        print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy20: {:.4f}, Time consumed:{:.2f}s'.format(
            epoch,
            test_loss / len(testloader.dataset),
            correct.float() / len(testloader.dataset),
            finish - start
        ))
        
    print()
    if only_coarse==False:
        wandb.log({"accuracy 100": correct.float() / len(testloader.dataset)})
        wandb.log({"accuracy 20": correctCoarse.float() / len(testloader.dataset)})
    else:
        wandb.log({"accuracy 20": correct.float() / len(testloader.dataset)})
    wandb.log({"testloss": test_loss / len(testloader.dataset)})

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(testloader.dataset), epoch)
        if only_coarse==False:
            writer.add_scalar('Test/Accuracy100', correct.float() / len(testloader.dataset), epoch)
            writer.add_scalar('Test/Accuracy20', correctCoarse.float() / len(testloader.dataset), epoch)
        else:
            writer.add_scalar('Test/Accuracy20', correct.float() / len(testloader.dataset), epoch)

    return correct.float() / len(testloader.dataset)

if __name__ == '__main__':
    wandb.init(project="two_steps", entity="hierarchical_classification")
    wandb.config = {"epochs": 200, "batch_size": 128}
    
    wandb.log({"fine set size": settings.COMPLEX_TRAINSET_SIZE})
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)
    net=net.cuda()

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    
    loss_function = nn.CrossEntropyLoss()
    loss_function2 = twoComponentLoss
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.PREMILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch, cifar100_training_loader)
        acc = eval_training(cifar100_test_loader, True, epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.PREMILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
        
    writer.close()
