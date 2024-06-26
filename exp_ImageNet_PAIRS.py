import numpy as np
import torch
import torch.nn as nn
import torchvision
from core import PAIRS, multi_evaluate, exp_aggregator, IdentityConv2d
from scipy.stats import norm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def experiment(num_neurons, batch_size):
    transforms = Compose([
        Resize(size=256),
        CenterCrop(size=(224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_dataset = torchvision.datasets.ImageNet(
        root='data/imagenet', split="val", transform=transforms,
    )

    indices = torch.randperm(len(base_dataset)).tolist()
    split_index = len(base_dataset) // 2

    train_dataset = torch.utils.data.Subset(base_dataset, indices[:split_index])
    val_dataset = torch.utils.data.Subset(base_dataset, indices[split_index:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = nn.Linear(3 * 224 * 224, num_neurons).to(device)

    with torch.no_grad():
        layer.weight.data.normal_()

    optimal_bias = norm.ppf(1 / batch_size) * np.sqrt(3 * 224 * 224)

    layer.bias.data.fill_(optimal_bias)

    model = IdentityConv2d(layer, 1000)

    PAIRS(
        layer=model.fc1,
        train_dataloader=train_loader,
        batch_size=batch_size,
        n_neurons=num_neurons,
    )

    return multi_evaluate(
        model=model,
        val_dataloader=val_loader,
        batch_size=batch_size,
        num_neurons=num_neurons,
        eval_iters=10
    )


def main():
    file_name = 'results_ImageNet_PAIRS.csv'
    torch.manual_seed(42)

    runs_per_setting = 10
    layer_sizes = [200, 500, 1000]
    batch_sizes = [20, 50, 100, 200]

    exp_aggregator(file_name, experiment, layer_sizes, batch_sizes, runs_per_setting)


if __name__ == "__main__":
    main()
