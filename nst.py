import torch
import torch.nn as nn
import torch.optim as optim

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
            if str(ln) in self.chosen_features:
                features.append(x)

        return features

model = VGG()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 356


loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

base_image = load_image('images/aizen_prof.jpg')
style_image = load_image('images/female_head_picasso.jpg')
gen_image = base_image.clone().requires_grad_(True)

total_steps = 6000
learning_rate = 1e-3
alpha = 1.0
beta = 0.01
optimizer = optim.Adam([gen_image] lr=learning_rate)

for step in range(total_steps):
    base_feats = model(base_image)
    style_feats = model(style_image)
    gen_feats = model(gen_image)





