
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import types
from model.classification_models import Pretrained_mnistNet


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss


def main():
    args = types.SimpleNamespace()
    # Training settings
 
  
    args.test_batch_size=1000
    args.epochs=25
    args.lr=1.0
    args.gamma=0.7
    args.no_cuda=False
    args.dry_run=False
    args.seed=1
    args.log_interval=10                      
    args.save_model=True 

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    test_small_loss = 10000000

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../MNIST_data/', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../MNIST_data/', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,batch_size= int(len(dataset1)/10),shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size= len(dataset2),shuffle=False)

    model = Pretrained_mnistNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        curr_test_loss = test(model, device, test_loader)
        scheduler.step()

        is_best = test_small_loss > curr_test_loss
        if args.save_model and is_best:
            torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()