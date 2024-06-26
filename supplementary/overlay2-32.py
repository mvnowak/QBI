import matplotlib.pyplot as plt
import torch
import numpy as np
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

num_images = 100


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


def main():
    data_path = ''

    if not data_path:
        print("Please specify the path to the ImageNet dataset")
        exit()

    images = load_imagenet_images(data_path)
    averaged_images = [images[0]]

    for i in range(1, 33):
        averaged_image = average_images(images[:i + 1])
        averaged_images.append(averaged_image)

    fig, axs = plt.subplots(1, 6, figsize=(15, 15))
    for i, ax in enumerate(axs.flat):
        if i == 0:
            ax.imshow(averaged_images[i])
            ax.set_title(f"n={i}")
            ax.axis('off')
        else:
            ax.imshow(averaged_images[2**i])
            ax.set_title(f"n={2**i}")
            ax.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()


if __name__ == "__main__":
    main()
