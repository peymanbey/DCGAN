import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10, MNIST
from torch.utils.data import DataLoader


def visual_data(batch, device):
    """
    Visualize a batch of image data

    Params
    --------------------------------
    batch:: a batch of images to visualize
            in numpy matrix format

    device:: cpu/gpu device used for computations
    """
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(
        batch[0].to(device)[:64],
        padding=2, normalize=True).cpu(), (1, 2, 0)))


def load_mnist(mnistFolder, tsfms, batchSize=64, numWorkers=0):
    """
    Load, download if necessary, the CIFAR10 data and
    return the iterable batch generator object

    Params
    --------------------------------
    cifarFolder:: folder to store/load the CIFAR10 data from
    tsfms:: transformations requaired to apply on data
    batchSize:: number of samples on each batch of the data
    numWorkers:: number of cpu cores used to load the data

    Return:
    --------------------------------
    trainLoader:: iterable training data batch generator object
    testLoader:: iterable test data batch generator object
    """
    trainData = MNIST(mnistFolder, download=True, train=True, transform=tsfms)
    trainLoader = DataLoader(
        trainData, batch_size=batchSize, num_workers=numWorkers)

    testData = MNIST(mnistFolder, download=True, train=False, transform=tsfms)
    testLoader = DataLoader(
        testData, batch_size=batchSize, num_workers=numWorkers)

    return trainLoader, testLoader


def load_cifar10(cifarFolder, tsfms, batchSize=64, numWorkers=0):
    """
    Load, download if necessary, the CIFAR10 data and
    return the iterable batch generator object

    Params
    --------------------------------
    cifarFolder:: folder to store/load the CIFAR10 data from
    tsfms:: transformations requaired to apply on data
    batchSize:: number of samples on each batch of the data
    numWorkers:: number of cpu cores used to load the data

    Return:
    --------------------------------
    trainLoader:: iterable training data batch generator object
    testLoader:: iterable test data batch generator object
    """
    trainData = CIFAR10(cifarFolder, download=True,
                        train=True, transform=tsfms)
    trainLoader = DataLoader(
        trainData, batch_size=batchSize, num_workers=numWorkers)

    testData = CIFAR10(cifarFolder, download=True,
                       train=False, transform=tsfms)
    testLoader = DataLoader(
        testData, batch_size=batchSize, num_workers=numWorkers)

    return trainLoader, testLoader
