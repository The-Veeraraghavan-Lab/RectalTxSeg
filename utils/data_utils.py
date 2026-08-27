# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.transforms import OneOf
from monai.data import load_decathlon_datalist
from monai.transforms import MapTransform, RandomizableTransform
from utils.monai_compat import channel_firstd


def _resolve_label_entry(item, args):
    label_key = getattr(args, "label_key", "label")
    fallback_label_key = getattr(args, "fallback_label_key", "label_plus")
    prefer_label_plus = getattr(args, "prefer_label_plus", False)

    label_value = item.get(label_key)
    fallback_value = item.get(fallback_label_key)

    if fallback_value:
        item["label"] = fallback_value
        return item

    if label_value:
        resolved_label = label_value
        if prefer_label_plus:
            resolved_label = resolved_label.replace("/label/", "/label_plus/")
            resolved_label = resolved_label.replace("\\label\\", "\\label_plus\\")
        item["label"] = resolved_label

    return item


def _normalize_datalist(items, args):
    return [_resolve_label_entry(dict(item), args) for item in items]


def _dataset_backend(args):
    if getattr(args, "use_normal_dataset", False):
        return "normal"
    return getattr(args, "dataset_backend", "cache")


def _percentile_bounds(args):
    lower = float(getattr(args, "percentile_lower", 0.0))
    upper = float(getattr(args, "percentile_upper", 99.5))
    if lower >= upper:
        raise ValueError(
            f"percentile_lower must be < percentile_upper, got {lower:g} >= {upper:g}"
        )
    return lower, upper


def intensity_cache_tag(args):
    mode = getattr(args, "intensity_mode", "fixed")
    if mode == "fixed":
        return f"hu{args.a_min:g}_{args.a_max:g}"
    if mode == "percentile":
        lower, upper = _percentile_bounds(args)
        return f"p{lower:g}_{upper:g}"
    raise ValueError(f"Unknown intensity_mode: {mode}")


def make_intensity_transform(keys, args):
    mode = getattr(args, "intensity_mode", "fixed")
    if mode == "fixed":
        return transforms.ScaleIntensityRanged(
            keys=keys,
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        )
    if mode == "percentile":
        lower, upper = _percentile_bounds(args)
        return transforms.ScaleIntensityRangePercentilesd(
            keys=keys,
            lower=lower,
            upper=upper,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        )
    raise ValueError(f"Unknown intensity_mode: {mode}")


def _persistent_cache_dir(args, split_name: str):
    root = getattr(args, "persistent_cache_root", None) or os.path.join(
        "saved_datasets", "persistent_cache"
    )
    json_stem = os.path.splitext(os.path.basename(args.json_list))[0]
    recipe = (
        f"{json_stem}_{getattr(args, 'train_key', 'training')}__{getattr(args, 'val_key', 'validation')}"
        f"_s{args.space_x:g}x{args.space_y:g}x{args.space_z:g}"
        f"_r{args.roi_x}x{args.roi_y}x{args.roi_z}"
        f"_{intensity_cache_tag(args)}"
        f"_{getattr(args, 'augmentation_mode', 'base')}"
    )
    return os.path.join(root, recipe, split_name)


def _make_dataset(items, transform, args, split_name: str, train: bool):
    backend = _dataset_backend(args)
    if backend == "normal":
        return data.Dataset(data=items, transform=transform)
    if backend == "persistent":
        return data.PersistentDataset(
            data=items,
            transform=transform,
            cache_dir=_persistent_cache_dir(args, split_name),
        )
    if train:
        return data.CacheDataset(
            data=items, transform=transform, cache_num=24, cache_rate=1.0, num_workers=args.workers
        )
    return data.Dataset(data=items, transform=transform)


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_loader_v2_mri(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            channel_firstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            make_intensity_transform(["image"], args),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            transforms.SpatialPadd(keys=["image","label"], spatial_size=(args.roi_x, args.roi_y, 0)),
            transforms.SpatialPadd(keys=["image","label"], spatial_size=(0, 0, args.roi_z), method = 'end'),

            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=getattr(args, "train_num_samples", 4),
                image_key="image",
                image_threshold=0,
            ),

            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),

            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            channel_firstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            make_intensity_transform(["image"], args),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            transforms.SpatialPadd(keys=["image","label"], spatial_size=(args.roi_x, args.roi_y, 0)),
            transforms.SpatialPadd(keys=["image","label"], spatial_size=(0, 0, args.roi_z), method = 'end'),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, getattr(args, "train_key", "training"), base_dir=data_dir)
    datalist = _normalize_datalist(datalist, args)
    train_ds = _make_dataset(datalist, train_transform, args, getattr(args, "train_key", "training"), train=True)
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_files = load_decathlon_datalist(datalist_json, True, getattr(args, "val_key", "validation"), base_dir=data_dir)
    val_files = _normalize_datalist(val_files, args)
    val_ds = _make_dataset(val_files, val_transform, args, getattr(args, "val_key", "validation"), train=False)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
    )
    loader = [train_loader, val_loader]

    return loader

def get_loader_v2_mri_adv(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    padding_transforms = [
        transforms.SpatialPadd(keys=["image","label"], spatial_size=(args.roi_x, args.roi_y, 0)),
        transforms.SpatialPadd(keys=["image","label"], spatial_size=(0, 0, args.roi_z), method = 'end'),
    ]
    train_transform_list = [
            transforms.LoadImaged(keys=["image", "label"]),
            channel_firstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),                                
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            make_intensity_transform(["image"], args),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            *padding_transforms,

            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=getattr(args, "train_num_samples", 4),
                image_key="image",
                image_threshold=0,
            ),
    ]
    train_transform_list.extend([
        transforms.RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.35, 0.35, 0.52),
            scale_range=(0.15, 0.15, 0.15),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),

        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3)
    ])

    if getattr(args, 'use_multiscale_aug', False):
        train_transform_list.append(
            transforms.RandZoomd(
                keys=["image", "label"], prob=0.3,
                min_zoom=(1.0, 1.0, 1.0), max_zoom=(1.4, 1.4, 1.0),
                mode=("trilinear", "nearest"), keep_size=True,
            )
        )

    # Tumor-specific augmentation (optional)
    if getattr(args, 'use_tumor_aug', False):
        train_transform_list.append(
            RandTumorIntensityd(
                keys=["image"], label_key="label",
                tumor_label=args.tumor_label,
                scale_range=(0.5, 1.5),
                shift_range=(-0.2, 0.2),
                prob=0.3,
            )
        )
    
    train_transform_list.extend([
         # Intensity augmentations
        transforms.RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
        transforms.RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5),
            prob=0.15,
        ),

        transforms.RandScaleIntensityd(keys="image", factors=0.2, prob=0.3),
        transforms.RandShiftIntensityd(keys="image", offsets=0.2, prob=0.1),
        transforms.RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7, 1.3)),
    ])
    train_transform_list.append(transforms.ToTensord(keys=["image", "label"]))
    train_transform = transforms.Compose(train_transform_list)

    
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            channel_firstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            make_intensity_transform(["image"], args),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            *padding_transforms,
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, getattr(args, "train_key", "training"), base_dir=data_dir)
    datalist = _normalize_datalist(datalist, args)
    train_ds = _make_dataset(datalist, train_transform, args, getattr(args, "train_key", "training"), train=True)
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_files = load_decathlon_datalist(datalist_json, True, getattr(args, "val_key", "validation"), base_dir=data_dir)
    val_files = _normalize_datalist(val_files, args)
    val_ds = _make_dataset(val_files, val_transform, args, getattr(args, "val_key", "validation"), train=False)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
    )
    loader = [train_loader, val_loader]

    return loader

class RandTumorIntensityd(RandomizableTransform, MapTransform):
    """Randomly modify intensity only within the tumor region."""
    
    def __init__(self, keys, label_key, tumor_label = 1, scale_range=(0.7, 1.3), shift_range=(-0.2, 0.2), prob=0.3,
     keep_first_original=True):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, prob)
        self.label_key = label_key
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.keep_first_original = keep_first_original
        self.sample_idx = 0
        self.tumor_label = tumor_label
        
    def randomize(self):
        super().randomize(None)
        if self._do_transform:
            self.scale = self.R.uniform(self.scale_range[0], self.scale_range[1])
            self.shift = self.R.uniform(self.shift_range[0], self.shift_range[1])
    
    def __call__(self, data):
        d = dict(data)
        
        # Check if this is a list (from RandCropByPosNegLabeld)
        if isinstance(d[self.keys[0]], list):
            for i in range(len(d[self.keys[0]])):
                # Skip first sample if keep_first_original
                if self.keep_first_original and i == 0:
                    continue
                    
                self.randomize()
                if not self._do_transform:
                    continue
                
                label = d[self.label_key][i]
                mask = label == self.tumor_label
                
                for key in self.keys:
                    img = d[key][i].clone() if isinstance(d[key][i], torch.Tensor) else d[key][i].copy()
                    img[mask] = img[mask] * self.scale + self.shift
                    d[key][i] = img
        else:
            # Single sample path (original behavior)
            self.randomize()
            if not self._do_transform:
                return d
            
            label = d[self.label_key]
            mask = label == self.tumor_label
            
            for key in self.keys:
                img = d[key].clone() if isinstance(d[key], torch.Tensor) else d[key].copy()
                img[mask] = img[mask] * self.scale + self.shift
                d[key] = img
        
        return d
