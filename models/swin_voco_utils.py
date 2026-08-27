"""MONAI SwinUNETR helpers for VoCo/Swin V2 fine-tuning."""

import inspect
import os

import torch
from monai.networks.nets import SwinUNETR

from models.effidec3d import SwinUNETR_EffiDec3D


def build_effidec3d(args):
    """SwinUNETR_EffiDec3D = the same SwinV2 encoder as swinv2, with the EffiDec3D
    channel-reduced / reduced-resolution decoder. Encoder SSL weights load the same
    way (into model.swinViT) via load_swin_encoder_pretrained."""
    return SwinUNETR_EffiDec3D(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        depths=tuple(args.swin_depths),
        num_heads=tuple(args.swin_num_heads),
        feature_size=args.feature_size,
        n_decoder_channels=args.n_decoder_channels,
        resolution_factor=args.resolution_factor,
        head_upsample=args.head_upsample,
        norm_name=args.norm_name,
        drop_rate=args.dropout_rate,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        spatial_dims=args.spatial_dims,
        use_v2=bool(args.swin_use_v2),
    )


def build_monai_swin_unetr(args):
    kwargs = dict(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        depths=tuple(args.swin_depths),
        num_heads=tuple(args.swin_num_heads),
        feature_size=args.feature_size,
        norm_name=args.norm_name,
        drop_rate=args.dropout_rate,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        spatial_dims=args.spatial_dims,
    )
    signature = inspect.signature(SwinUNETR.__init__).parameters
    if "use_v2" in signature:
        kwargs["use_v2"] = bool(args.swin_use_v2)
    elif args.swin_use_v2:
        raise RuntimeError("Installed MONAI SwinUNETR does not expose use_v2.")
    return SwinUNETR(**kwargs)


def resolve_checkpoint_path(path, default_dir="pretrained_models"):
    if not path or os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(default_dir, path)


def extract_swin_encoder_state(checkpoint):
    if isinstance(checkpoint, dict) and "teacher" in checkpoint:
        state = checkpoint["teacher"]
    elif isinstance(checkpoint, dict) and "student" in checkpoint:
        state = checkpoint["student"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    prefixes = (
        "module.backbone.swinViT.",
        "backbone.swinViT.",
        "module.swinViT.",
        "swinViT.",
        "module.",
    )
    encoder_prefixes = (
        "patch_embed.",
        "layers1.",
        "layers2.",
        "layers3.",
        "layers4.",
        "layers1c.",
        "layers2c.",
        "layers3c.",
        "layers4c.",
    )
    out = {}
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        if new_key.startswith(encoder_prefixes):
            out[new_key] = value

    return {
        key.replace("mlp.fc1", "mlp.linear1").replace("mlp.fc2", "mlp.linear2"): value
        for key, value in out.items()
    }


def load_swin_encoder_pretrained(model, checkpoint_path, strict=False):
    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    swin_state = extract_swin_encoder_state(checkpoint)
    if not swin_state:
        raise ValueError(f"No SwinUNETR encoder weights found in {checkpoint_path}")
    msg = model.swinViT.load_state_dict(swin_state, strict=strict)
    print(
        f"Loaded {len(swin_state)} SwinUNETR encoder keys from {checkpoint_path}; "
        f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}"
    )
    if msg.missing_keys:
        print("First missing keys:", msg.missing_keys[:5])
    if msg.unexpected_keys:
        print("First unexpected keys:", msg.unexpected_keys[:5])
    return msg
