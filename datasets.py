import json
from abc import abstractstaticmethod
from pathlib import Path
from typing import Union, Callable, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, default_collate
from PIL import Image


@dataclass
class CaptionExample:
    id: int
    image_id: int
    image_file: torch.Tensor
    caption: str


class CaptionDataset(Dataset):

    def __init__(
        self,
        split: str,
        data_dir: Union[str, Path] = None,
        image_transform: Callable[[Image.Image], torch.Tensor] = None,
        text_transform: Callable[[List[str]], torch.Tensor] = None,
    ):
        super().__init__()
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.samples = self.load_data(split, data_dir)

    def __getitem__(self, index):
        sample = self.samples[index]

        image = Image.open(sample.image_file).convert('RGB')
        if self.image_transform is not None:
            image = self.image_transform(image)
        
        return {
            'id': sample.id,
            'image_id': sample.image_id,
            'image': image,
            'caption': sample.caption,
        }

    def __len__(self):
        return len(self.samples)

    def collate_fn(self, batch):
        elem = batch[0]
        transposed = {key: [d[key] for d in batch] for key in elem.keys()}

        ids = default_collate(transposed['id'])
        image_ids = default_collate(transposed['image_id'])

        images = transposed['image']
        if isinstance(images[0], torch.Tensor):
            images = default_collate(images)

        captions = transposed['caption']
        if self.text_transform is not None:
            captions = self.text_transform(captions).input_ids
        
        return {
            'id': ids,
            'image_id': image_ids,
            'image': images,
            'caption': captions
        }

    @abstractstaticmethod
    def load_data(split: str, data_dir: Union[str, Path]):
        pass


class COCODataset(CaptionDataset):

    @staticmethod
    def load_data(split, data_dir=None):
        assert split in ['train', 'val'], f"Invalid split: {split}"
        if data_dir is None:
            data_dir = Path.home() / 'scratch/datasets/coco'
        else:
            data_dir = Path(data_dir)

        with open(data_dir / 'annotations' / f'captions_{split}2014.json') as f:
            data = json.load(f)
        images = {img['id']: img for img in data['images']}
        captions = data['annotations']

        samples = []        
        for cap in captions:
            caption_id = cap['id']
            image_id = cap['image_id']
            caption = cap['caption']
            image_file = data_dir / 'images' / f'{split}2014' / images[image_id]['file_name']
            samples.append(CaptionExample(caption_id, image_id, image_file, caption))

        return samples
