import os
import torch
import torchvision
import torchvision.models as models
import torch
import torch.nn as nn

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

def model_fn(model_dir):
    
    model = net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model