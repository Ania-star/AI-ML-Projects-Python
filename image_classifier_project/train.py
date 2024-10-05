#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#                                                                             
# PROGRAMMER: Anna Bajszczak
# DATE CREATED: 9/29/2024                                   
# REVISED DATE: 
# PURPOSE: Train a neural network on a dataset of images with options to adjust
#          model architecture, hyperparameters, and GPU training using command 
#          line arguments. If the user fails to provide some or all of the inputs, 
#          then the default values are used for the missing inputs. Command Line Arguments:.
#     1. Data directory as --data_dir with default value 'dir'
#     2. Model Architecture as --arch with default value 'vgg'
#     3. Learning rate as --learning_rate with default value 0.001
#     4. Number of hidden units as --hidden_units with default value 200
#     5. Number of epochs as --epochs with default value 8
#     6. Enable GPU for training as --gpu with default value 'False'


# Imports python modules

import argparse
import json
import torch
from torchvision import models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def arg_parser():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description="Train a neural network")
    
    # Create 10 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('--train_dir', type=str, default='data/train', help='path to folder that contains training images')
    parser.add_argument('--validate_dir', type=str, default='data/valid', help='path to folder that contains validation images')
    parser.add_argument('--test_dir', type=str, default='data/test', help='path to folder that contains testing images')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture (vgg16 or resnet18)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_units_1', type=int, default=200, help='Number of hidden units for 1st hidden layer in the classifier')
    parser.add_argument('--hidden_units_2', type=int, default=150 , help='Number of hidden units for 2nd hidden layer in the classifier')
    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=8)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help='Directory to save the trained model checkpoint')
    args = parser.parse_args()
    return args

def train_model(args):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # Load the dataset
    data_transforms = {
        'train': transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
        'validate': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
        'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    dataset = { 'train':ImageFolder(root=args.train_dir, transform=data_transforms['train']), 
               'validate':ImageFolder(root=args.validate_dir, transform=data_transforms['validate']),
               'test':ImageFolder(root=args.test_dir, transform=data_transforms['test'])
    }
    loader = {
        'train':DataLoader(dataset['train'], batch_size=64, shuffle=True),
        'validate': DataLoader(dataset['validate'], batch_size=64, shuffle=True),
        'test': DataLoader(dataset['test'], batch_size=64, shuffle=True)
        }
    
    # Create the class_to_idx mapping
    class_to_idx = dataset['train'].class_to_idx
   
    # Choose the model architecture
    if args.arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        input_size = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(input_size, args.hidden_units_1),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(args.hidden_units_1, args.hidden_units_2),
            nn.ReLU(),
            nn.Linear(args.hidden_units_2, len(dataset['train'].classes)),
            nn.LogSoftmax(dim=1))
    elif args.arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  
        input_size = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(input_size, args.hidden_units_1),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(args.hidden_units_1, args.hidden_units_2),
            nn.ReLU(),
            nn.Linear(args.hidden_units_2, len(dataset['train'].classes)),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError("Unsupported architecture. Choose either 'vgg16' or 'resnet18'.")

    print('Model successfully loaded with new architecture')
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    
    if args.arch == 'vgg16':
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif args.arch == 'resnet18':
        for param in model.fc.parameters():
            param.requires_grad = True

    model.to(device)

    # Create the optimizer for the classifier only
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    criterion = nn.NLLLoss()


    total_loss = 0.0  # tracking loss
    steps = 0 # tracking steps 
    print_every = 1

    # Training loop
    for epoch in range(args.epochs):
        model.train()  # Set the model to training mode
        
        for inputs, labels in loader['train']: #loop thru data for images
            

            steps +=1
            inputs, labels = inputs.to(device), labels.to(device) # if available move images and lables to GPU for training
            
        
            optimizer.zero_grad() # zeroing gradients for each batch
        
            outputs = model(inputs) # passing images through the model
            loss = criterion (outputs, labels) # calculating loss
            loss.backward() # computing gradiant of all loss
            optimizer.step() # updating weights

            total_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0.0
                accuracy = 0.0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in loader['validate']:
                        
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_class = ps.topk(1, dim=1)[1]
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{args.epochs}.. "
                    f"Train loss: {total_loss/print_every:.3f}.. "
                    f"Validation loss: {test_loss/len(loader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(loader):.3f}")
                total_loss = 0
                model.train()
            
        
    #Testing
    with torch.no_grad():
        model.eval()
        accuracy = 0.0
    
        for inputs, labels in loader['test']:
            inputs, labels = inputs.to(device), labels.to(device)
        
            logps = model(inputs)
        
            # Calculate accuracy
            ps = torch.exp(logps)
            top_class = ps.topk(1, dim=1)[1]
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    # Calculate and print the average accuracy
    accuracy = accuracy / len(loader['test']) * 100  
    print(f"Testing complete. Accuracy: {accuracy:.2f}%")

    checkpoint={ #creating dictionary
    'structure': args.arch,  
    'classifier' : model.classifier,
    'epochs': args.epochs,  
    'state_dict': model.state_dict(),  
    'class_to_idx': class_to_idx,  
    'optimizer_state_dict': optimizer.state_dict()  }
    # Save the model
    torch.save(checkpoint, f"{args.save_dir}/checkpoint.pth")
    print("Model saved!")

if __name__ == "__main__":
    args = arg_parser()
    train_model(args)