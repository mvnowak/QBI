import csv
import os.path
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm


class IMDBModel(nn.Module):
    """
    The model used to acquire results listed in Table 2, operating on the IMDB sentiment analysis dataset.
    """

    def __init__(self, fc1):
        super(IMDBModel, self).__init__()
        self.fc1 = fc1
        self.classification = nn.Linear(self.fc1.out_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.classification(x)
        return x


class IMDBDataset(Dataset):
    """
    The dataset used to acquire results listed in Table 2, operating on the IMDB sentiment analysis dataset.
    Embeddings can be fetched from the dataset for performance reasons, as we only simulate a single training step.
    """

    def __init__(self, data, tokenizer, device):
        self.device = device
        self.data = data
        self.tokenizer = tokenizer
        self.embedding = nn.Embedding(num_embeddings=10_000, embedding_dim=250).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=250,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_tensors="pt"
        )

        return self.embedding(encoding["input_ids"].flatten().to(self.device)), torch.tensor(label).to(self.device)


class IdentityConv2d(nn.Module):
    """
    Model architecture used for all demo notebooks and image experiments.
    Convolutional layers are maliciously initialized to identity functions by default.
    """

    def __init__(self, fc1, num_classes):
        super(IdentityConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc1 = fc1
        self.classification = nn.Linear(self.fc1.out_features, num_classes)
        self.activation_counts = None

        """
        See Appendix C.3
        """
        for conv in [self.conv1, self.conv2, self.conv3]:
            for i in range(3):
                conv.weight.data[i, :, :, :] = 0
                conv.weight.data[i, i, 1, 1] = 1

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x.flatten(start_dim=1))

        """
        This simulates the client-side injection of AGGP, as described in Section 5, storing activation
        counts during the forward pass
        """
        with torch.no_grad():
            self.activation_counts = (x > 0).sum(dim=0)

        x = torch.relu(x)
        x = self.classification(x)
        return x


def multi_evaluate(model, val_dataloader, batch_size, num_neurons, eval_iters, batch_norm=False, layer_norm=False,
                   shape=(3, 224, 224)):
    """
    Evaluates metrics A, P, and R (see Equations 8 through 10) for a given model, batch size, and number of neurons.
    Averages results over `eval_iters` iterations.
    """
    if batch_norm:
        norm = nn.BatchNorm2d([*shape][0])
    elif layer_norm:
        norm = nn.LayerNorm(normalized_shape=[*shape])
    layer = model.fc1
    device = layer.weight.device
    total_A, total_P, total_R = 0, 0, 0
    progressbar = tqdm(total=eval_iters)
    for i, (batch, _) in enumerate(val_dataloader):
        progressbar.update(1)
        if i == eval_iters:
            break
        if batch_norm or layer_norm:
            batch = norm(batch)
        data_points = batch.flatten(start_dim=1).to(device)
        A, P, R = evaluate(layer, data_points)
        total_A += A
        total_P += P
        total_R += R
        progressbar.set_description(f"Unseen data: A: {A} P: {P} R: {R}")
    progressbar.close()
    total_A = round(total_A / eval_iters / num_neurons, 2) * 100
    total_P = round(total_P / eval_iters / num_neurons, 2) * 100
    total_R = round(total_R / eval_iters / batch_size, 2) * 100

    print(f"Eval Result for Setting ({num_neurons}, {batch_size}): A: {total_A}% P: {total_P}% R: {total_R}%")
    return total_A, total_P, total_R


def evaluate(layer, data):
    """
    Determines metrics A, P, and R (see Equations 8 through 10) for a given layer and single data batch.
    """
    with torch.no_grad():

        outputs = layer(data)
        P = 0
        A = 0
        unique_datapoints = set()
        for i in range(outputs.shape[1]):
            neuron_outputs = outputs[:, i]
            active_data_points = (neuron_outputs > 0).nonzero().squeeze(1)
            if len(active_data_points) == 1:
                P += 1
                unique_datapoints.add(active_data_points[0].item())
            if len(active_data_points) > 0:
                A += 1
        R = len(unique_datapoints)
        return A, P, R


def PAIRS(layer, train_dataloader, batch_size, n_neurons):
    """
    Implementation of the PAIRS algorithm, as described in Section 4, Algorithm 1
    :param layer: the layer to be maliciously initialized
    :param train_dataloader: train_loader containing auxiliary data from the target domain
    :param batch_size: the batch size that will be used on the client side
    :param n_neurons: the number of neurons in the layer
    """
    print(f"Running tuning for scenario ({n_neurons}, {batch_size})")
    num_iterations = batch_size * n_neurons * 5
    num_batches = (n_neurons // batch_size) if n_neurons % batch_size == 0 else (n_neurons // batch_size) + 1
    device = layer.weight.device

    for batch_idx, (batch, _) in enumerate(train_dataloader):
        batch = batch.flatten(start_dim=1).to(device)

        frozen_datapoints = set()

        last_neuron_index = min(n_neurons, (batch_idx + 1) * batch_size)
        neurons_per_batch = last_neuron_index - (batch_idx * batch_size)

        if neurons_per_batch <= 0:
            break

        progress_bar = tqdm(total=num_iterations)

        retries = 0
        no_progress = 0
        for ni in range(batch_idx * batch_size, last_neuron_index):
            while retries < num_iterations:
                if no_progress > 2000:
                    break
                progress_bar.set_description(
                    f"[{batch_idx + 1}/{num_batches}]Num of frozen datapoints: {len(frozen_datapoints)}")
                progress_bar.update(1)

                # initial pass to determine activation pattern
                with torch.no_grad():
                    neuron_outputs = layer(batch)[:, ni]

                active_data_points = (neuron_outputs > 0).nonzero().squeeze(1)
                if len(active_data_points) != 1 or active_data_points[0].item() in frozen_datapoints:
                    layer.weight.data[ni].normal_()
                    retries += 1
                    no_progress += 1
                    continue

                frozen_datapoints.add(active_data_points[0].item())
                no_progress = 0
                progress_bar.set_description(
                    f"[{batch_idx + 1}/{num_batches}]Num of frozen datapoints: {len(frozen_datapoints)}")
                break
            if no_progress > 2000:
                break

        progress_bar.close()


def exp_aggregator(file_name, experiment, layer_sizes, batch_sizes, runs_per_setting):
    """
    Aggregates results from multiple runs of an experiment and writes them to a CSV file.
    :param file_name: the name of the file to write the results to
    :param experiment: experiment function that will be called with different parameters
    :param layer_sizes: all possible layer sizes that will be evaluated
    :param batch_sizes: all possible batch sizes that will be evaluated
    :param runs_per_setting: number of runs per setting (95% confidence interval will be calculated based on this)
    :return:
    """
    file_name = os.path.join("results", file_name)
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["Layer Size", "Batch Size", "A", "A Error Margin", "P", "P Error Margin", "R", "R Error Margin"])

        for (n_neurons, batch_size) in list(product(layer_sizes, batch_sizes)):
            As, Ps, Rs = [], [], []
            for _ in range(runs_per_setting):
                A, P, R = experiment(n_neurons, batch_size)
                As.append(A)
                Ps.append(P)
                Rs.append(R)

            mean_A = sum(As) / runs_per_setting
            mean_P = sum(Ps) / runs_per_setting
            mean_R = sum(Rs) / runs_per_setting

            std_A = torch.std(torch.tensor(As))
            std_P = torch.std(torch.tensor(Ps))
            std_R = torch.std(torch.tensor(Rs))

            error_margin_A = 1.96 * std_A / torch.sqrt(torch.tensor(runs_per_setting))
            error_margin_P = 1.96 * std_P / torch.sqrt(torch.tensor(runs_per_setting))
            error_margin_R = 1.96 * std_R / torch.sqrt(torch.tensor(runs_per_setting))

            writer.writerow([n_neurons, batch_size, round(mean_A, 2), round(error_margin_A.item(), 2), round(mean_P, 2),
                             round(error_margin_P.item(), 2), round(mean_R, 2), round(error_margin_R.item(), 2)])
            csvfile.flush()
