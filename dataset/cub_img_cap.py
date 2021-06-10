from typing import Optional, Callable, Tuple, Any, List
from torchvision.datasets.folder import default_loader

import os
from transformers import RobertaTokenizer
import numpy as np
import torchvision.datasets as datasets


class ImgCapCUB200(datasets.VisionDataset):
  def __init__(self, root: str, tokenizer_path: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    super().__init__(root, transform=transform, target_transform=target_transform)
    self.class_idx = self.get_class_idx(root)
    self.classes = list(self.class_idx.keys())
    self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    self.data = self.parse_data_file(root, train)

    self.loader = default_loader

  def __getitem__(self, index: int) -> Any:
    path, input_id, mask, target = self.data[index]

    img = self.loader(path)
    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None and target is not None:
      target = self.target_transform(target)

    return img, input_id, mask, target

  def __len__(self) -> int:
    return len(self.data)

  def parse_data_file(self, root: str, train: bool) -> List[Tuple[str, int]]:
    with open(os.path.join(root, "CUB_200_2011/images.txt")) as f:
      lines = []
      paths = []
      for line in f.readlines():
        _, line = line.strip().split()
        path = os.path.join(root, "CUB_200_2011/images", line)
        lines.append(self.load_text(path.replace("CUB_200_2011","text_c10").replace("images/", "").replace(".jpg", ".txt")))
        paths.append(path)
      tokens = self.tokenizer(lines, return_tensors="pt", padding=True)
    with open(os.path.join(root, "CUB_200_2011/train_test_split.txt")) as f:
      splits = f.readlines()

    with open(os.path.join(root, "CUB_200_2011/image_class_labels.txt")) as f:
      labels = f.readlines()

    data_list = []
    for path, input_id, mask, split, label in zip(paths, tokens.input_ids, tokens.attention_mask, splits, labels):

      _, split = split.strip().split()
      _, label = label.strip().split()

      if int(split) == int(train):

        data_list.append((path, input_id, mask, int(label)-1))

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

  def load_text(self, path, nth_len=3):
    with open(os.path.join(path)) as f:
      lines = []
      lengths = []
      for line in f.readlines():
        lines.append(line)
        line = list(filter(lambda x: len(x)!=1, line.split(" ")))
        lengths.append(len(line))
      input_txt = lines[lengths.index(sorted(lengths)[-nth_len])]
    return input_txt