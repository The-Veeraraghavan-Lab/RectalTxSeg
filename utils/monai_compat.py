"""Small MONAI API compatibility helpers."""

import inspect

from monai import transforms


def channel_firstd(keys):
    """Add a channel dimension for images that do not store one on disk."""
    if hasattr(transforms, "EnsureChannelFirstd"):
        signature = inspect.signature(transforms.EnsureChannelFirstd.__init__).parameters
        if "channel_dim" in signature:
            return transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel")
        return transforms.EnsureChannelFirstd(keys=keys)
    return transforms.AddChanneld(keys=keys)
