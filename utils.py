""" helper function

author baiyu
"""
import os
import sys
import re
import datetime
from conf import settings

import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net

def change_labels_to_coarse(dataset): 
    newset=list(dataset)
    for i in range(len(newset)):
        buf=list(newset[i])
        buf[1]=superclass[buf[1]] 
        newset[i]=tuple(buf)
    
    return newset

def twoComponentLoss(outputs, class_labels, superclass_labels):
    loss = nn.CrossEntropyLoss()
    func=max
    print("\n\nclass_labels: ", class_labels)
    print("\n\noutputs: ", outputs)
    coarse = []
    for i in range(len(outputs)):
        coarse.append([])
        
    for i in range(len(outputs)): #по максимуму/сумме выходов для классов определяются выходы для суперклассов
        coarse[i].append(func([outputs[i][72], outputs[i][4], outputs[i][95], outputs[i][30], outputs[i][55]]))
        coarse[i].append(func([outputs[i][73], outputs[i][32], outputs[i][67], outputs[i][91], outputs[i][1]]))
        coarse[i].append(func([outputs[i][92], outputs[i][70], outputs[i][82], outputs[i][54], outputs[i][62]]))
        coarse[i].append(func([outputs[i][16], outputs[i][61], outputs[i][9], outputs[i][10], outputs[i][28]]))
        coarse[i].append(func([outputs[i][51], outputs[i][0], outputs[i][53], outputs[i][57], outputs[i][83]]))
        coarse[i].append(func([outputs[i][40], outputs[i][39], outputs[i][22], outputs[i][87], outputs[i][86]]))
        coarse[i].append(func([outputs[i][20], outputs[i][25], outputs[i][94], outputs[i][84], outputs[i][5]]))
        coarse[i].append(func([outputs[i][14], outputs[i][24], outputs[i][6], outputs[i][7], outputs[i][18]]))
        coarse[i].append(func([outputs[i][43], outputs[i][97], outputs[i][42], outputs[i][3], outputs[i][88]]))
        coarse[i].append(func([outputs[i][37], outputs[i][17], outputs[i][76], outputs[i][12], outputs[i][68]]))
        coarse[i].append(func([outputs[i][49], outputs[i][33], outputs[i][71], outputs[i][23], outputs[i][60]]))
        coarse[i].append(func([outputs[i][15], outputs[i][21], outputs[i][19], outputs[i][31], outputs[i][38]]))
        coarse[i].append(func([outputs[i][75], outputs[i][63], outputs[i][64], outputs[i][66], outputs[i][34]]))
        coarse[i].append(func([outputs[i][77], outputs[i][26], outputs[i][45], outputs[i][99], outputs[i][79]]))
        coarse[i].append(func([outputs[i][11], outputs[i][2], outputs[i][35], outputs[i][46], outputs[i][98]]))
        coarse[i].append(func([outputs[i][29], outputs[i][93], outputs[i][27], outputs[i][78], outputs[i][44]]))
        coarse[i].append(func([outputs[i][65], outputs[i][50], outputs[i][74], outputs[i][36], outputs[i][80]]))
        coarse[i].append(func([outputs[i][56], outputs[i][52], outputs[i][47], outputs[i][59], outputs[i][96]]))
        coarse[i].append(func([outputs[i][8], outputs[i][58], outputs[i][90], outputs[i][13], outputs[i][48]]))
        coarse[i].append(func([outputs[i][81], outputs[i][69], outputs[i][41], outputs[i][89], outputs[i][85]]))
        
    print("\n\ncoarse",coarse)
    #l1=loss(torch.tensor(coarse).cuda(), superclass_labels)
    l1=F.cross_entropy(torch.tensor(coarse).cuda(), superclass_labels)
    print("\n\nl1", l1)
    
    mask = class_labels < 0
    indices = torch.nonzero(mask) 
    
    for i in len(class_labels):
      if i in indices:
        outputs = torch.cat((outputs[:i,:], outputs[i:,:]), dim = 0)
        class_labels=torch.cat((class_labels[:i], class_labels[i:]), dim = 0)
      else:
        i=i+1
    
    print("\n\noutputs after deleting: ", outputs)
    
    #outs1=torch.tensor(outputs).cuda()
    #classes1=torch.tensor(class_labels).cuda()
    
    #print("outs1: ", outs1)
    
    print("shape of outputs: ", outputs.size())
    print("shape of class_labels: ", class_labels.size())
    
    #l2=loss(outputs, class_labels)   
    #print(l2)
    
    l2=F.cross_entropy(torch.tensor(outputs).cuda(), class_labels)
    print(l2)
    
    return 0.7*l1+0.3*l2
    
  
def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    
    
    cifar100_trainset1, cifar100_trainset2 = torch.utils.data.random_split(cifar100_training, [settings.COMPLEX_TRAINSET_SIZE, 50000-settings.COMPLEX_TRAINSET_SIZE], generator=torch.Generator().manual_seed(0))
  
    cifar100_trainset1=list(cifar100_trainset1)
    for i in range(len(cifar100_trainset1)):
      cifar100_trainset1[i]=list(cifar100_trainset1[i])
      cifar100_trainset1[i].append(cifar100_trainset1[i][1])
      cifar100_trainset1[i]=tuple(cifar100_trainset1[i])
      
    cifar100_trainset1_1 = change_labels_to_coarse(cifar100_trainset1)
    print("\nfine train loader labels examples:")
    for i in range(10):
      print(cifar100_trainset1_1[i][1], ', ', cifar100_trainset1_1[i][2])
     
    cifar100_trainset2=list(cifar100_trainset2)
    for i in range(len(cifar100_trainset2)):
      cifar100_trainset2[i]=list(cifar100_trainset2[i])
      cifar100_trainset2[i].append(-1)
      cifar100_trainset2[i]=tuple(cifar100_trainset2[i])
      
    cifar100_trainset2_1 = change_labels_to_coarse(cifar100_trainset2)
    print("\ncoarse train loader labels examples:")
    for i in range(10):
      print(cifar100_trainset2_1[i][1], ', ', cifar100_trainset2_1[i][2])
    
    cifar100_global_dataset=cifar100_trainset2_1+cifar100_trainset1_1
    print("\nglobal dataset labels:")
    for i in range(20):
        print(cifar100_global_dataset[i][1], '-->', cifar100_global_dataset[i][2])
                                                
    cifar100_training_loader = DataLoader(
        cifar100_global_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_splitted_dataloaders(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_trainset1, cifar100_trainset2 = torch.utils.data.random_split(cifar100_training, [settings.COMPLEX_TRAINSET_SIZE, 50000-settings.COMPLEX_TRAINSET_SIZE], generator=torch.Generator().manual_seed(0))
    print('First (coarse) dataset size:', len(cifar100_trainset2)) #trainset2 - with coarse 
    print('Second (fine) dataset size:', len(cifar100_trainset1)) #trainset1 - with classes
    
    cifar100_trainset2_1=change_labels_to_coarse(cifar100_trainset2)
    
    print("\nCoarse dataset labels:")
    for i in range(20):
      print(cifar100_trainset2[i][1], "-->", cifar100_trainset2_1[i][1])
      
    print("\nFine dataset labels:")
    for i in range(20):
      print(cifar100_trainset1[i][1])
    print()
   
    cifar100_training_loader1 = DataLoader(
          cifar100_trainset1, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    cifar100_training_loader2 = DataLoader(
          cifar100_trainset2_1, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader2, cifar100_training_loader1

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def get_coarse_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    cifar100_test1 = change_labels_to_coarse(cifar100_test)
    
    print("Generated coarse set. Labels:")
    for i in range(10):
      print(cifar100_test[i][1], "-->", cifar100_test1[i][1])
    print()
    
    cifar100_test_loader = DataLoader(
        cifar100_test1, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]
