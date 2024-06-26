import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from scipy.stats import norm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from core import PAIRS, multi_evaluate, exp_aggregator, IMDBModel, IMDBDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def experiment(num_neurons, batch_size):
    imdb_dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")

    train_dataset = IMDBDataset(imdb_dataset["train"], tokenizer, device)
    test_dataset = IMDBDataset(imdb_dataset["test"], tokenizer, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    layer = nn.Linear(250 * 250, num_neurons).to(device)

    model = IMDBModel(layer)

    optimal_bias = norm.ppf(1 / batch_size) * np.sqrt(250 * 250)
    with torch.no_grad():
        layer.weight.data.normal_()
        layer.bias.data.fill_(optimal_bias)

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
    file_name = "results_IMDB_PAIRS.csv"
    torch.manual_seed(42)

    runs_per_setting = 10
    layer_sizes = [1000]
    batch_sizes = [20, 50, 100, 200]

    exp_aggregator(file_name, experiment, layer_sizes, batch_sizes, runs_per_setting)


if __name__ == "__main__":
    main()
