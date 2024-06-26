import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset

from core import multi_evaluate, exp_aggregator


class SyntheticModel(nn.Module):
    def __init__(self, fc1):
        super(SyntheticModel, self).__init__()
        self.fc1 = fc1

    def forward(self, x):
        return self.fc1(x)


class RandomDataset(Dataset):
    def __init__(self, shape):
        self.shape = shape

    def __len__(self):
        return 1 << 20

    def __getitem__(self, idx):
        data = torch.randn(self.shape)
        label = torch.zeros(1)
        return data, label


def custom_collate(batch):
    data, labels = zip(*batch)
    return torch.stack(data), torch.stack(labels)


def experiment(num_neurons, batch_size):
    test_dataset = RandomDataset((3, 32, 32))

    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = nn.Linear(3 * 32 * 32, num_neurons).to(device)

    with torch.no_grad():
        layer.weight.data.normal_()

    optimal_bias = norm.ppf(1 / batch_size) * np.sqrt(3 * 32 * 32)
    layer.bias.data.fill_(optimal_bias)

    model = SyntheticModel(layer)

    return multi_evaluate(
        model=model,
        val_dataloader=val_loader,
        batch_size=batch_size,
        num_neurons=num_neurons,
        eval_iters=10
    )


def main():
    file_name = 'results_random.csv'
    torch.manual_seed(42)

    runs_per_setting = 100
    layer_sizes = [200, 500, 1000]
    batch_sizes = [20, 50, 100, 200]

    exp_aggregator(file_name, experiment, layer_sizes, batch_sizes, runs_per_setting)


if __name__ == "__main__":
    main()
