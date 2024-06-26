import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import norm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

from core import multi_evaluate, exp_aggregator, IdentityConv2d


def experiment(num_neurons, batch_size):
    transforms = Compose([
        ToTensor()
    ])

    base_dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, transform=transforms, download=True
    )

    indices = torch.randperm(len(base_dataset)).tolist()
    split_index = len(base_dataset) // 2

    val_dataset = torch.utils.data.Subset(base_dataset, indices[split_index:])

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = nn.Linear(3 * 32 * 32, num_neurons).to(device)

    with torch.no_grad():
        layer.weight.data.normal_()

    optimal_bias = norm.ppf(1 / batch_size) * np.sqrt(3 * 32 * 32)
    layer.bias.data.fill_(optimal_bias)

    model = IdentityConv2d(layer, 10)

    return multi_evaluate(
        model=model,
        val_dataloader=val_loader,
        batch_size=batch_size,
        num_neurons=num_neurons,
        eval_iters=10,
        layer_norm=True,
        shape=(3, 32, 32)
    )


def main():
    file_name = 'results_CIFAR-10_QBI_Layernorm.csv'
    torch.manual_seed(42)

    runs_per_setting = 10
    layer_sizes = [200, 500, 1000]
    batch_sizes = [20, 50, 100, 200]

    exp_aggregator(file_name, experiment, layer_sizes, batch_sizes, runs_per_setting)


if __name__ == "__main__":
    main()
