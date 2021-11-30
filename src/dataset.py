import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import requests
import zipfile

from tqdm.auto import tqdm

class CustomDataset(object):
    def __init__(self, dataset_name="COCO", transforms=transforms.ToTensor(), root="dataset"):
        """
            Dataset class for VAE training.

            args.
                dataset: "COCO" or "MNIST".
                transforms: A transforms object. the way you want to tranform your images.
                root: The path where the dataset will be stored.
        """
        self.dataset_name = dataset_name
        self.transforms = transforms
        self.root = root
        self.trainset, self.testset = self.__get_dataset(self.dataset_name, self.root)

    def __get_dataset(self, dataset_name, root="dataset"):
        def get_coco_dataset(root, transforms):
            def download(url, root):
                name = os.path.join(root, url.split('/')[-1])
                print(f'Downloading file from {url} to {name}')
                
                with open(name, 'wb') as f:
                    res = requests.get(url, stream=True)
                    length = res.headers.get('Content-Length')
                    
                    if length is None:
                        f.write(res.content)
                    else:
                        for data in tqdm(res.iter_content(chunk_size=4096), total=int(int(length) / 4096)):
                            f.write(data)
                return name

            train_dir = os.path.join(root, "test")
            if not os.path.isdir(train_dir):
                os.makedirs(train_dir)
                # download
                train_archive = download('http://images.cocodataset.org/zips/test2017.zip', train_dir)
                # unzip
                with zipfile.ZipFile(train_archive, 'r') as f:
                    f.extractall(train_dir)

            test_dir = os.path.join(root, "val")
            if not os.path.isdir(test_dir):
                os.makedirs(test_dir)
                # download
                test_archive = download('http://images.cocodataset.org/zips/val2017.zip', test_dir)
                # unzip
                with zipfile.ZipFile(test_archive, 'r') as f:
                    f.extractall(test_dir)

            trainset = datasets.ImageFolder(root=train_dir, transform=transforms)
            testset = datasets.ImageFolder(root=test_dir, transform=transforms)
            return trainset, testset

        def get_mnist_dataset(root, transforms):
            trainset = datasets.MNIST(root, train=True, download=True, transform=transforms)
            testset = datasets.MNIST(root, train=False, download=True, transform=transforms)
            return trainset, testset

        if dataset_name == "COCO":
            return get_coco_dataset(os.path.join(root, "coco"), self.transforms)
        elif dataset_name == "MNIST":
            return get_mnist_dataset(os.path.join(root, "mnist"), self.transforms)

    def get_dataloader(self, is_train=True, **kwargs):
        if is_train:
            return DataLoader(self.trainset, **kwargs)
        else:
            return DataLoader(self.testset, **kwargs)

