from __future__ import print_function

import os
import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from utils import visualize_stn
from model import Net
from train import train
from test import test


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True




def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Working on GPU/CPU mode
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(device)
    
    # Training dataset
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Test dataset
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


    model = Net(enable_coordconv=True).to(device)
    print(model)


    # Optimizers
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               betas=(args.beta1,args.beta2),
                               eps=1e-8,
                               weight_decay=1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)


    # TensorBoard
    #save_path = 'logs/MNIST_Experiment'
    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)
    #writer = SummaryWriter(save_path)

    print("Start training ...")
    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(model, optimizer, train_loader, device, epoch)
        test_loss, acc = test(model, test_loader, device)
        
        # logging losses and accuracy
        #writer.add_scalar('training loss', train_loss, epoch)
        #writer.add_scalar('testing loss', test_loss, epoch)
        #writer.add_scalar('Accuracy', acc, epoch)
    
    # Visualize the STN transformation on some input batch
    visualize_stn(model, test_loader, device, enable_coordconv=True)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='STN model testing.')
    
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--use_cuda', default=True, type=bool, help='enable cuda')
    
    # Dataset setting Parameters
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')

    # training Settings
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--optimizer', default='SGD', type=str, help='Specify the optimizer.')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')
    
    args = parser.parse_args()

    main(args)
