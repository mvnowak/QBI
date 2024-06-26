import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import norm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from core import multi_evaluate, exp_aggregator, IdentityConv2d


def experiment(num_neurons, batch_size):
    transforms = Compose([
        Resize(size=256),
        CenterCrop(size=(224, 224)),
        ToTensor()
    ])

    base_dataset = torchvision.datasets.ImageNet(
        root='data/imagenet', split="val", transform=transforms,
    )

    val_loader = DataLoader(base_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = nn.Linear(3 * 224 * 224, num_neurons).to(device)

    with torch.no_grad():
        layer.weight.data.normal_()

    optimal_bias = norm.ppf(1 / batch_size) * np.sqrt(3 * 224 * 224)

    layer.bias.data.fill_(optimal_bias)
    model = IdentityConv2d(layer, 1000)

    return multi_evaluate(
        model=model,
        val_dataloader=val_loader,
        batch_size=batch_size,
        num_neurons=num_neurons,
        eval_iters=10,
        batch_norm=True
    )


def main():
    file_name = 'results_ImageNet_QBI_Batchnorm.csv'
    torch.manual_seed(42)

    runs_per_setting = 10
    layer_sizes = [200, 500, 1000]
    batch_sizes = [20, 50, 100, 200]

    exp_aggregator(file_name, experiment, layer_sizes, batch_sizes, runs_per_setting)


if __name__ == "__main__":
    main()
