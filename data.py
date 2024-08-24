import os
import torch
import torchvision

from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR10, MNIST, LFWPeople

import torchvision.transforms as transforms


mnist_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels = 3),
    transforms.Resize((64, 64)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: (x - 0.5) * 2)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    # transforms.Grayscale(num_output_channels = 3),
    transforms.Resize((64, 64)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: (x - 0.5) * 2)
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def prepare_dataloader(args):
    if args['dataset_name'] == "cifar10":
        print("With CIFAR10")
        cifar10_train_dataset = CIFAR10(root = "./data/cifar10", train = True, transform = transform, download = True)
        cifar10_test_dataset = CIFAR10(root = "./data/cifar10", train = False, transform = transform, download = True)
        
        train_dataloader = DataLoader(cifar10_train_dataset, batch_size = args['batch_size'], shuffle = True, drop_last = True)
        test_dataloader = DataLoader(cifar10_test_dataset, batch_size = args['batch_size'], shuffle = False, drop_last = True)
        
    elif args['dataset_name'] == "mnist":
        print("With MNIST")
        mnist_train_dataset = MNIST(root = './data', train = True, transform = mnist_transform, download = True)
        mnist_test_dataset = MNIST(root = './data', train = False, transform = mnist_transform, download = True)
        
        train_dataloader = DataLoader(mnist_train_dataset, batch_size = args['batch_size'], shuffle = True, drop_last = True)
        test_dataloader = DataLoader(mnist_test_dataset, batch_size = args['batch_size'], shuffle = False, drop_last = True)
        
    elif args['dataset_name'] == 'lfw':
        lfw_train_data = LFWPeople(root = "./data", split = 'train', image_set = 'original', 
                      transform = transform, download = True)

        lfw_test_data = LFWPeople(root = "./data", split = 'test', image_set = 'original', 
                      transform = transform, download = True)
        
        train_dataloader = DataLoader(lfw_train_data, batch_size = args['batch_size'], shuffle = True, drop_last = True)
        test_dataloader = DataLoader(lfw_test_data, batch_size = args['batch_size'], shuffle = False, drop_last = True)  
        
    else:
        print("check proper db name it must be from mnist, cifar10, stl10, fashionmnist")
        return None, None
        
    return train_dataloader, test_dataloader


# # lfw_train_data = LFWPeople(root = "./data", split = 'train', image_set = 'original', 
# #                       transform = transform, download = True)

# # lfw_test_data = LFWPeople(root = "./data", split = 'test', image_set = 'original', 
# #                       transform = transform, download = True)

# # print("With CIFAR10")
# # cifar10_train_dataset = CIFAR10(root = "./data", train = True, transform = transform, download = True)
# # cifar10_test_dataset = CIFAR10(root = "./data", train = False, transform = transform, download = True)


# print("With MNIST")
# mnist_train_dataset = MNIST(root = './data', train = True, transform = mnist_transform, download = True)
# mnist_test_dataset = MNIST(root = './data', train = False, transform = mnist_transform, download = True)

# # lfw_train_dataloader = DataLoader(lfw_train_data, batch_size = 64, shuffle = True, drop_last = True)
# # lfw_test_dataloader = DataLoader(lfw_test_data, batch_size = 603, shuffle = False, drop_last = True)


# # split = 0.1

# # num_samples_train = int(split * len(lfw_train_data))
# # train_indices = torch.randperm(len(lfw_train_data))[:num_samples_train]

# # num_samples_valid = int(split * len(lfw_train_data))
# # valid_indices = torch.randperm(len(lfw_train_data))[:num_samples_valid]

# # num_samples_test = int(split * len(lfw_test_data))
# # test_indices = torch.randperm(len(lfw_test_data))[:num_samples_test]

# # split_tarin_data = Subset(lfw_train_data, train_indices)
# # split_valid_data = Subset(lfw_train_data, valid_indices)
# # split_test_data = Subset(lfw_test_data, test_indices)


# train_dataloader = DataLoader(mnist_train_dataset, batch_size = 64, shuffle = True, drop_last = True)
# test_dataloader = DataLoader(mnist_test_dataset, batch_size = 64, shuffle = False, drop_last = True)