import torch
import torch.nn as nn

import torchvision.transforms as transforms

from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image

from PIL import Image

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:29]

    def forward(self, x):
        features = []

        for ln, layer in enumerate(self.model):
            x = layer(x)
            if ln in self.chosen_features:
                features.append(x)

        return features

model = VGG()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 356


loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

base_image = load_image('images/aizen_prof.png')
style_image = load_image('images/female_head_picasso.jpg')

print(base_image.shape, style_image.shape)

gen_image = base_image.clone().requires_grad_(True)

base_feats = model(base_image)
print(base_feats)


