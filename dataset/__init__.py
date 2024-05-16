import os
from torch.utils.data import DataLoader
from torchvision import transforms

from .lesions_dataset import MSLesionDataset
from .siamese_dataset import MSSiameseLesionDataset
from .utils import Normalize, NormalizeSiamese


def get_train_loaders(train_file,valid_file, data_directory, batch_size):
    train_csv = data_directory+train_file
    valid_csv = data_directory+valid_file
    train_dataset = MSLesionDataset(csv_file=train_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               Normalize()
                                           ]))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


    valid_dataset = MSLesionDataset(csv_file=valid_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               Normalize()
                                           ]))

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    return train_dataloader, valid_dataloader

def get_test_loaders(test_file, data_directory):
    test_csv = data_directory+test_file
    test_dataset = MSSiameseLesionDataset(csv_file=test_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               NormalizeSiamese()
                                           ]))

    test_dataloader = DataLoader(test_dataset, batch_size=1,
                        shuffle=False, num_workers=0)


    return test_dataloader

def get_siamese_train_loaders(train_file,valid_file, data_directory, batch_size=4):
    train_csv = os.path.join(data_directory,train_file)
    valid_csv = os.path.join(data_directory,valid_file)
    train_dataset = MSSiameseLesionDataset(csv_file=train_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               NormalizeSiamese()
                                           ]))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


    valid_dataset = MSSiameseLesionDataset(csv_file=valid_csv,
                                    root_dir=data_directory,
                               transform = transforms.Compose([
                                               NormalizeSiamese()
                                           ]))

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

    return train_dataloader, valid_dataloader
