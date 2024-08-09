"""
This python script converts the network into Script Module
"""
import torch
import torchvision.models
from torch import jit
from torchvision import models

# Download and load the pre-trained model
model = models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)

# Set upgrading the gradients to False
# for param in model.parameters():
#	param.requires_grad = False

example_input = torch.rand(1, 3, 224, 224)
print(model(example_input).shape)

script_module = torch.jit.trace(model, example_input)
script_module.save('AlexNet_jit_model.pt')

