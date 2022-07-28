import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.models as models

from torchvision import save_image

from PIL import Image

model = models.vgg19(pretrained=True).features
print(model)


