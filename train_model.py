#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import smdebug.pytorch as smd
import argparse
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
import torch.nn.functional as F
import os
import sys
import json
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(model, train_loader, criterion, optimizer, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    hook.set_mode(smd.modes.TRAIN) 
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            logger.info(
                "Train : [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return model    
    
    pass
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
    nn.Linear(num_features, 133))
    
    return model

def create_data_loaders(data_dir, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("Create train data loader")
    cus_transform=transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize((224,224))
    ])
    data = torchvision.datasets.ImageFolder(root=data_dir, transform=cus_transform)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train_loader = create_data_loaders(args.train_data_dir, args.batch_size)
    test_loader = create_data_loaders(args.test_data_dir, args.batch_size)
    
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    # refer to https://sagemaker.readthedocs.io/en/stable/overview.html#prepare-a-training-script
    
    parser.add_argument(
        "--train-data-dir", # the actual variable is train_data_dir
        type=str,
        default=os.environ['SM_CHANNEL_TRAIN'],
        metavar="TDD",
        help="Training data directory",
    )
    parser.add_argument(
        "--test-data-dir",
        type=str,
        default=os.environ['SM_CHANNEL_TEST'],
        metavar="EDD",
        help="Test data directory",
    )
    parser.add_argument(
        "--val-data-dir",
        type=str,
        default=os.environ['SM_CHANNEL_VAL'],
        metavar="VDD",
        help="Test data directory",
    )
    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

    args=parser.parse_args()
    
    main(args)