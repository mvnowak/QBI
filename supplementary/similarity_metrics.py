import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

num_experiments = 100
num_images_per_exp = 30
num_images = num_experiments * num_images_per_exp


def load_imagenet_images(data_path):
    transforms = Compose([
        Resize(size=256),
        CenterCrop(size=(224, 224)),
        ToTensor(),
    ])

    base_dataset = torchvision.datasets.ImageNet(
        root=data_path, split="val", transform=transforms,
    )

    indices = torch.randperm(len(base_dataset)).tolist()
    base_dataset = torch.utils.data.Subset(base_dataset, indices[:num_images])

    images = []
    for i in range(num_images):
        image, _ = base_dataset[i]
        image_array = image.numpy().transpose(1, 2, 0)
        images.append(image_array)
    return images


def average_images(images):
    return np.mean(images, axis=0)


def compute_metrics(original_image, averaged_image):
    psnr = peak_signal_noise_ratio(original_image, averaged_image)
    ssim = structural_similarity(original_image, averaged_image, multichannel=True, channel_axis=2, data_range=1)
    l1_distance = np.mean(np.abs(original_image - averaged_image))
    return psnr, ssim, l1_distance


def main():
    # Parameters
    data_path = ''

    if not data_path:
        print("Please specify the path to the ImageNet dataset")
        exit()

    # Create a list to store the metrics for each experiment
    metrics = {
        'PSNR': [],
        'SSIM': [],
        'L1': [],
    }

    for e in range(num_experiments):
        print(e)

        # Load multiple ImageNet images
        images = load_imagenet_images(data_path)

        # Create a list to store the averaged images and metrics for this experiment
        averaged_images = [images[0]]
        experiment_metrics = {
            'PSNR': [],
            'SSIM': [],
            'L1': [],
        }

        # Average the image with 1 more, 2 more, etc. up to 25 images
        for i in range(1, 26):
            averaged_image = average_images(images[:i + 1])
            averaged_images.append(averaged_image)
            psnr, ssim, l1_distance = compute_metrics(images[0], averaged_image)
            experiment_metrics['PSNR'].append(psnr)
            experiment_metrics['SSIM'].append(ssim)
            experiment_metrics['L1'].append(l1_distance)

        # Add the experiment metrics to the overall metrics
        for metric, values in experiment_metrics.items():
            metrics[metric].append(values)

    # Compute the average metrics across all experiments
    average_metrics = {}
    for metric, values in metrics.items():
        average_metrics[metric] = np.mean(values, axis=0)

    # Plot the average metrics
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.set_xlabel('Number of Averaged Images')
    ax1.set_ylabel('SSIM, L1')
    ax2.set_ylabel('PSNR')

    for metric, values in average_metrics.items():
        if metric == 'PSNR':
            ax2.plot(range(1, 26), values, label=metric, color='r')
        else:
            ax1.plot(range(1, 26), values, label=metric)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
