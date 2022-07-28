import torch
import torch.nn as nn

import torchvision.transforms as transforms
import torchvision.models as models

from torchvision.utils import save_image

from PIL import Image

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    print(device)
    return image.to(device)

model = models.vgg19(pretrained=True).features
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
image_size = 356

loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

base_image = load_image('images/aizen_prof.png')
style_image = load_image('images/female_head_picasso.jpg')



