import urllib.request
import zipfile
import os
import warnings

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.transforms import RandomRotation, Normalize, Pad, Resize, ToTensor, Compose, InterpolationMode

import pytorch_lightning as pl

import numpy as np

from PIL import Image

from .utils import ZeroRotation

warnings.filterwarnings("ignore", message="The given NumPy array is not writeable")

class MNISTRotModule(pl.LightningDataModule):
    def __init__(self,
                 dir: str = "./data",
                 batch_size: int = 32,
                 num_workers: int = 0,
                 upsample: bool = True,
                 normalize: bool = True,
                 augment: bool = True,
                 pad: bool = False,
                 shuffle: bool = True,
                 validation_size = 2000):
        super().__init__()
        self.dir = dir
        self.train_file = f"{dir}/mnist_rot_trainval.npz"
        self.test_file = f"{dir}/mnist_rot_test.npz"
        self.batch_size = batch_size
        self.upsample = upsample
        self.normalize = normalize
        self.augment = augment
        self.num_workers = num_workers
        self.validation_size = validation_size
        self.pad = pad
        self.shuffle = shuffle
        self.img_size = 29 if self.pad else 28
        self.dims = (1, self.img_size, self.img_size)
        self.num_batches = (12000 - self.validation_size) // self.batch_size

    def prepare_data(self):
        if os.path.isfile(self.train_file) and os.path.isfile(self.test_file):
            return
        
        os.makedirs(self.dir, exist_ok=True)

        zip_path = f"{self.dir}/mnist_rot.zip"
        urllib.request.urlretrieve("http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip", zip_path)

        with zipfile.ZipFile(zip_path, 'r') as file:
            file.extractall(self.dir)
        
        data = np.loadtxt(f"{self.dir}/mnist_all_rotation_normalized_float_train_valid.amat", delimiter=' ')
        images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        labels = data[:, -1].astype(np.int64)
        np.savez(f"{self.dir}/mnist_rot_trainval.npz", images=images, labels=labels)

        data = np.loadtxt(f"{self.dir}/mnist_all_rotation_normalized_float_test.amat", delimiter=' ')
        images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        labels = data[:, -1].astype(np.int64)
        np.savez(f"{self.dir}/mnist_rot_test.npz", images=images, labels=labels)

        assert os.path.isfile(self.train_file) and os.path.isfile(self.test_file)
    
    def setup(self, stage : str = None):
        
        train_transforms = []
        test_transforms = []
        if self.pad:
            # images are padded to have shape 29x29.
            # this allows to use odd-size filters with stride 2 when downsampling a feature map in the model
            pad = Pad((0, 0, 1, 1), fill=0)
            train_transforms.append(pad)
            test_transforms.append(pad)

        # to reduce interpolation artifacts (e.g. when testing the model on rotated images),
        # we upsample an image by a factor of 3, rotate it and finally downsample it again
        if self.upsample and self.augment:
            resize1 = Resize(3 * self.img_size)
            train_transforms.append(resize1)
            # we also resize the test set, to create the same kind of artifacts there
            test_transforms.append(resize1)
        

        if self.augment:
            train_transforms.append(RandomRotation(180, interpolation=InterpolationMode.BILINEAR))
            # On the test set, we do a rotation by 0 degrees, just to create the artifacts
            test_transforms.append(ZeroRotation(interpolation=InterpolationMode.BILINEAR))

        if self.upsample and self.augment:
            resize2 = Resize(self.img_size)
            train_transforms.append(resize2)
            test_transforms.append(resize2)

        totensor = ToTensor()
        train_transforms.append(totensor)
        test_transforms.append(totensor)

        if self.normalize:
            # magic numbers are the mean and std of the training set
            # can be computed with calculate_dataset_stats.py,
            # IF THE NORMALIZATION IS DISABLED
            if self.augment:
                mean = 0.1299176
                std = 0.27349371
            else:
                mean = 0.1299588
                std = 0.29697534
            normalize = Normalize((mean, ), (std, ))
            train_transforms.append(normalize)
            test_transforms.append(normalize)

        train_transform = Compose(train_transforms)
        test_transform = Compose(test_transforms)
            
        if stage == "fit" or stage is None:
            dataset = MNISTRotDataset(data_dir=self.dir, mode='train', transform=train_transform)
            if self.validation_size:
                self.mnist_train, self.mnist_val = random_split(
                    dataset,
                    [12000 - self.validation_size, self.validation_size]
                )
                # HACK: maybe we should implement the random split ourselves instead?
                self.mnist_val.dataset.transform = test_transform
            else:
                self.mnist_train = dataset
                # use the test set as a validation set, so we already get some feedback
                # before the end of a run.
                # main.py should ensure that this isn't used for early stopping or checkpointing
                test_set = MNISTRotDataset(data_dir=self.dir, mode='test', transform=test_transform)
                # using the entire test set would be overkill though, we only use a random subset
                self.mnist_val = Subset(test_set, np.random.choice(len(test_set), 2000, replace=False))

        if stage == "test" or stage is None:
            self.mnist_test = MNISTRotDataset(data_dir=self.dir, mode='test', transform=test_transform)
    
    def train_dataloader(self):
        return DataLoader(self.mnist_train,
                          pin_memory=True,
                          drop_last=True,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=self.shuffle)

    def val_dataloader(self):
        if hasattr(self, "mnist_val"):
            return DataLoader(self.mnist_val,
                              pin_memory=True,
                              num_workers=self.num_workers,
                              batch_size=self.batch_size)
        else:
            return None

    def test_dataloader(self):
        return DataLoader(self.mnist_test,
                          pin_memory=True,
                          num_workers=self.num_workers,
                          batch_size=self.batch_size)


class MNISTRotDataset(Dataset):
    
    def __init__(self, mode, data_dir = "./data", transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = f"{data_dir}/mnist_rot_trainval.npz"
        else:
            file = f"{data_dir}/mnist_rot_test.npz"

        data = np.load(file)
        self.images = data['images'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        
        self.transform = transform
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)


