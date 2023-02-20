#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms

import time
import argparse
import os
import subprocess
import logging
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# takes a model and a testing data loader and will get the test accuray/loss of the model
def test(model, testloader, criterion):
    print("Validation")

    device=torch.device("cuda:0")
    model=model.to(device)
    model.eval()
    
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    
    sample_size = 0.02*len(testloader.dataset)
    
    with torch.no_grad():
        for i, data in enumerate(testloader):
            counter+=1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            
            if counter > sample_size:
                break  
            
        # loss and accuracy for the complete epoch
        epoch_loss = valid_running_loss / sample_size
        epoch_acc = 100. * (valid_running_correct / sample_size)        

        logger.info(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                epoch_loss, valid_running_correct, sample_size, epoch_acc
            )
        )
        
        return epoch_loss, epoch_acc

    


def train(args, model, trainloader, criterion, optimizer):
    # take a model and data loaders for training and will get train the model
    device=torch.device("cuda:0")
    print(f"Training on {device}")

    model = model.to(device)
    model.train()
    train_running_loss = 0
    train_running_correct = 0
    counter = 0
        
    sample_size = 0.05*len(trainloader.dataset)
        
    for (i, data) in enumerate(trainloader):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds==labels).sum().item()
        
        loss.backward() # backprop
        optimizer.step() # update the optimizer parameters

        if counter > sample_size:
            break  
                
    epoch_loss = train_running_loss / sample_size
    epoch_acc = 100. * (train_running_correct / sample_size)
    
    return epoch_loss, epoch_acc
    
def net():
    # function that initializes model
    # using a pretrained model
    net = models.resnet18(pretrained=True)
    
    for param in net.parameters():
        param.requires_grad = False # don't update weights in the pretrained model
        
    num_features = net.fc.in_features
    net.fc = nn.Sequential(nn.Linear(num_features, 133))
    
    return net
        
    
def main(args):
    # Initialize a model by calling the net function
    model=net()
    
    # TODO: Create your loss and optimizer
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args["lr"], momentum=0.9)
    
    
    # start training your model
    # get training data from S3
    root_dir = "dogImages"
    imgSize = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((imgSize,imgSize)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((imgSize,imgSize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    
    # download images from S3
    remote_folder = "s3://deeplearning-project/dogImages"
    local_path="dogImages"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    subprocess.run(['aws', 's3', 'sync', remote_folder, local_path])
    
    
    trainset = datasets.ImageFolder(f"{root_dir}/train", transform=train_transform)
    validset = datasets.ImageFolder(f"{root_dir}/valid", transform=test_transform)
    testset = datasets.ImageFolder(f"{root_dir}/test", transform=test_transform)
    
    
    
    dataset_size = len(trainset)
    print(f"Total number of training images: {dataset_size}")
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args["batch_size"], shuffle=True, num_workers=2)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args["test_batch_size"], shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args["test_batch_size"], shuffle=False, num_workers=2)
    
                                             
    trainloss, validloss = [], []
    trainacc, validacc = [], []
                                             
                                             
    for epoch in range(args["epochs"]):
        print(f"[INFO]: Epoch {epoch} of {args['epochs']}")
        train_epoch_loss, train_epoch_acc = train(args, model, trainloader, loss_criterion, optimizer)        
        valid_epoch_loss, valid_epoch_acc = test(model, validloader, loss_criterion)
                                             
        trainloss.append(train_epoch_loss)
        validloss.append(valid_epoch_loss)
        trainacc.append(train_epoch_acc)
        validacc.append(valid_epoch_acc)

        logger.info(f"Epoch {epoch} of {args['epochs']}. Train loss: {train_epoch_loss:.3f}. Train accuracy: {train_epoch_acc:.3f}")
        
        # print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        # print(f"Validation loss: {valid_epoch_loss:.3f}, valid acc: {valid_epoch_acc:.3f}")
                                             
        print('-'*50)
        time.sleep(3)    
    
    # save the trained model
    # logger.info("Saving the model")
    # torch.save(model.state_dict(), "/opt/ml/model/model.pth")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    # Specify any training args
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    

    args=parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}:{value}")
    
    main(vars(args))