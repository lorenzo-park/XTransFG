from typing import Optional, Callable, Tuple, Any, List
from torchvision.datasets.folder import default_loader

import os
import torch
import numpy as np
import torchvision.datasets as datasets


class CUB200(datasets.VisionDataset):
  def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    super().__init__(root, transform=transform, target_transform=target_transform)
    self.class_idx = self.get_class_idx(root)
    self.classes = list(self.class_idx.keys())
    self.data = self.parse_data_file(root, train)
    np.ones(len(self.data) - int())
    np.zeros(int(len(self.data) * 0.))

    self.loader = default_loader

  def __getitem__(self, index: int) -> Any:
    path, target = self.data[index]
    img = self.loader(path)
    txt = torch.load(path.replace(".jpg", ".txt.pt"))
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None and target is not None:
      target = self.target_transform(target)
    return img, txt, target

  def __len__(self) -> int:
    return len(self.data)

  def parse_data_file(self, root: str, train: bool) -> List[Tuple[str, int]]:
    with open(os.path.join(root, "CUB_200_2011/images.txt")) as f:
      lines = f.readlines()

    with open(os.path.join(root, "CUB_200_2011/train_test_split.txt")) as f:
      splits = f.readlines()

    with open(os.path.join(root, "CUB_200_2011/image_class_labels.txt")) as f:
      labels = f.readlines()

    data_list = []
    for line, split, label in zip(lines, splits, labels):
      _, line = line.strip().split()
      _, split = split.strip().split()
      _, label = label.strip().split()

      if int(split) == int(train):
        data_list.append((os.path.join(root, "CUB_200_2011/images", line), label))

    return data_list

  def get_class_idx(self, root: str):
    class_idx = {}
    with open(os.path.join(root, "CUB_200_2011/classes.txt")) as f:
      for line in f.readlines():
        idx, classname = line.strip().split(" ")
        class_idx[classname] = idx
    return class_idx

  @property
  def num_classes(self) -> int:
    """Number of classes"""
    return len(self.classes)