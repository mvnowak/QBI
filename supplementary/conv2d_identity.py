import torch
import torch.nn as nn

import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop

data_path = ''

if not data_path:
    print("Please specify the path to the ImageNet dataset")
    exit()


class IdentityConv2d(nn.Module):
    def __init__(self):
        super(IdentityConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        for conv in [self.conv1, self.conv2, self.conv3]:
            for i in range(3):
                conv.weight.data[i, :, :, :] = 0
                conv.weight.data[i, i, 1, 1] = 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


transforms = Compose([
    Resize(size=256),
    CenterCrop(size=(224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

base_dataset = torchvision.datasets.ImageNet(
    root=data_path, split="val", transform=transforms,
)

train_loader = DataLoader(base_dataset, batch_size=2, shuffle=True)

input_tensor, label = next(iter(train_loader))

identity = IdentityConv2d()
output_tensor = identity(input_tensor)

assert torch.allclose(input_tensor, output_tensor), "Input and output tensors are not identical"
print(f"Input and output tensor equal: {torch.equal(input_tensor, output_tensor)}")
