"""
WaterBirds dataset
Returns (x, y, g) on enumerating where g is the group label
Reference: https://github.com/ssagawa/overparam_spur_corr/blob/master/data/cub_dataset.py
"""

import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_transform_cub(train, augment_data):
  scale = 256.0 / 224.0
  target_resolution = (224, 224)
  assert target_resolution is not None

  if (not train) or (not augment_data):
    # Resizes the image to a slightly larger square then crops the center.
    transform = transforms.Compose([
      transforms.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
      transforms.CenterCrop(target_resolution),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  else:
    transform = transforms.Compose([
      transforms.RandomResizedCrop(
        target_resolution,
        scale=(0.7, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=2),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  return transform


class WaterBirds(Dataset):

  def __init__(self, root_dir, split, augment=None):
    # self.root_dir = root_dir
    self.data_dir = os.path.join(root_dir, 'waterbirds')

    # self.data_dir = os.path.join(
    #   self.root_dir,
    #   '_'.join([self.target_name] + self.confounder_names))

    if not os.path.exists(self.data_dir):
      raise ValueError(
        f'{self.data_dir} does not exist yet. Please generate the dataset first.')

    # Read in metadata
    self.metadata_df = pd.read_csv(
      os.path.join(self.data_dir, 'metadata.csv'))

    # Get the y values
    self.y_array = self.metadata_df['y'].values
    self.n_classes = 2

    # We only support one confounder for CUB for now
    self.confounder_array = self.metadata_df['place'].values
    self.n_confounders = 1
    # Map to groups
    self.n_groups = pow(2, 2)
    self.group_array = (self.y_array * (self.n_groups / 2) + self.confounder_array).astype('int')

    # Extract filenames and splits
    self.filename_array = self.metadata_df['img_filename'].values
    self.split_array = self.metadata_df['split'].values
    self.split_dict = {
      'train': 0,
      'val': 1,
      'test': 2
    }

    self.split = split
    split_mask = self.split_array == self.split_dict[split]
    indices = np.where(split_mask)[0]
    self.filename_array = self.filename_array[indices]
    self.y_array = self.y_array[indices]
    self.group_array = self.group_array[indices]

    augment_data = (augment is not None)
    self.augment = augment
    self.train_transform = get_transform_cub(
      train=True,
      augment_data=augment_data)
    self.eval_transform = get_transform_cub(
      train=False,
      augment_data=augment_data)

  def __len__(self):
    return len(self.filename_array)

  def __getitem__(self, idx):
    y = self.y_array[idx]
    g = self.group_array[idx]

    img_filename = os.path.join(
      self.data_dir,
      self.filename_array[idx])
    img = Image.open(img_filename).convert('RGB')
    # Figure out split and transform accordingly
    if self.augment == 'train':
      img = self.train_transform(img)
    elif self.augment == 'test':
      img = self.eval_transform(img)
    x = img

    return x, y, g