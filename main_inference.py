import argparse
import os
import re

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from monai import data, transforms
from monai.data import MetaTensor, load_decathlon_datalist, decollate_batch
from monai.inferers import sliding_window_inference
from models import smit, configs_smit
from models.swin_voco_utils import build_monai_swin_unetr, build_effidec3d
from utils.data_utils import make_intensity_transform
from utils.monai_compat import channel_firstd

from tqdm import tqdm


def get_key_name_part(path, min_len=4):
    """Extract meaningful identifier from file path."""
    fname = os.path.basename(path)
    stem = fname.split('.')[0]
    parent = os.path.basename(os.path.dirname(path))

    if re.fullmatch(r"\d{4,}", stem):
        return stem

    throwaway = {"ctimg", "mrimg", "scan", "img", "image"}
    if len(stem) < min_len or stem.lower() in throwaway:
        return parent

    return stem


def make_seg_filename(path):
    """Generate segmentation filename from input path."""
    key = get_key_name_part(path)
    return f"{key}_seg.nii.gz"


def list_of_strings(arg):
    """Parse comma-separated string into list."""
    return arg.split(',')


def setup_argparser():
    """Setup argument parser with all configuration options."""
    parser = argparse.ArgumentParser(description="Unified SMIT segmentation pipeline with optional TTA and distributed inference")
    
    # Data arguments
    parser.add_argument("--data_dir", default=None, type=str, help="dataset directory")
    parser.add_argument("--json_list", default=None, type=str, help="dataset json file")
    parser.add_argument("--datasets", default=None, type=list_of_strings, 
                        help="comma-separated list of datasets to pull from json file")
    parser.add_argument("--results_dir", default='results', type=str, help="main output directory")
    parser.add_argument("--output_dir", default=None, type=str, help="secondary output directory")

    # Model arguments
    parser.add_argument("--model_name", default='smit', type=str, help="architecture key that selects the model to build")
    parser.add_argument("--pred_subdir", default=None, type=str,
                        help="name of the prediction subfolder (defaults to model_name); "
                             "set this to distinguish runs that share an architecture, e.g. voxelfox vs voco")
    parser.add_argument("--pretrained_model_path", default=None, type=str, 
                        help="path to pretrained model checkpoint")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
    parser.add_argument("--norm_name", default="batch", type=str, help="normalization name")
    parser.add_argument("--use_upernet", action="store_true", help="Use UPERNET decoder")
    parser.add_argument("--feature_size", default=48, type=int, help="MONAI SwinUNETR feature size")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension")
    parser.add_argument("--swin_depths", default=(2, 2, 2, 2), type=int, nargs=4,
                        help="MONAI SwinUNETR stage depths for swinv2 (SwinV2 SwinUNETR)")
    parser.add_argument("--swin_num_heads", default=(3, 6, 12, 24), type=int, nargs=4,
                        help="MONAI SwinUNETR attention heads for swinv2 (SwinV2 SwinUNETR)")
    parser.add_argument("--swin_use_v2", action="store_true",
                        help="Use MONAI SwinUNETR V2 blocks for swinv2 (SwinV2 SwinUNETR)")
    # --- LoRA / LoRA-Ensemble: MUST mirror the training run so the checkpoint loads (same wrapping). ---
    parser.add_argument("--use_lora", action="store_true",
                        help="checkpoint was trained with LoRA/LoRA-Ensemble; inject the same wrapping before load")
    parser.add_argument("--lora_rank", default=4, type=int, help="LoRA rank - MUST match the training run")
    parser.add_argument("--lora_members", default=1, type=int,
                        help="LoRA members - MUST match training (>1 = LoRA-Ensemble; outputs reduced softmax-then-mean)")
    parser.add_argument("--allow_roi_mismatch", action="store_true",
                        help="permit inference ROI to differ from the checkpoint's training ROI (normally a bug; this loads fine but corrupts results)")
    parser.add_argument("--decoder_ensemble", action="store_true",
                        help="checkpoint is the LoRA + decoder-ensemble upper-bound variant; rebuild that structure before load. MUST match training.")
    parser.add_argument("--lora_decoder_ensemble", action="store_true",
                        help="checkpoint is the paper-closest LoRA-Ensemble variant with per-member LoRA adapters and decoders.")
    # --- effidec3d (must match the values used at training time to rebuild the arch) ---
    parser.add_argument("--n_decoder_channels", default=48, type=int,
                        help="effidec3d: fixed channel width across all decoder stages")
    parser.add_argument("--resolution_factor", default=2, type=int,
                        help="effidec3d: coarsest output stride (1=full res, 2=half res, ...)")
    parser.add_argument("--head_upsample", default="trilinear", type=str,
                        choices=["none", "trilinear", "upconv", "upconv_refine"],
                        help="effidec3d: how the reduced-res output is restored to full res")


    # Preprocessing arguments
    parser.add_argument("--intensity_mode", "--intensity-mode", dest="intensity_mode",
                        default="fixed", choices=["fixed", "percentile"],
                        help="image intensity scaling: fixed uses --a_min/--a_max; "
                             "percentile rescales each volume using --percentile_lower/--percentile_upper")
    parser.add_argument("--a_min", default=0.0, type=float, help="lower bound for --intensity_mode fixed")
    parser.add_argument("--a_max", default=800.0, type=float, help="upper bound for --intensity_mode fixed")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--percentile_lower", "--percentile-lower", dest="percentile_lower",
                        default=0.0, type=float,
                        help="lower percentile for --intensity_mode percentile")
    parser.add_argument("--percentile_upper", "--percentile-upper", dest="percentile_upper",
                        default=99.5, type=float,
                        help="upper percentile for --intensity_mode percentile")
    parser.add_argument("--space_x", default=1.0, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.0, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=1.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    
    # Inference arguments
    parser.add_argument("--infer_overlap", default=0.5, type=float, 
                        help="sliding window inference overlap")
    parser.add_argument("--sw_batch_size", default=16, type=int, 
                        help="sliding window batch size")
    parser.add_argument("--use_tta", action="store_true", 
                        help="use test-time augmentation (horizontal flip)")
    
    # Orientation argument
    parser.add_argument("--skip_orientation", action="store_true",
                        help="skip Orientationd transform (for datasets already in correct orientation)")
    
    # Post-processing arguments
    parser.add_argument("--postproc", default="standard", 
                        choices=['standard', 'conf_threshold'],
                        help="post-processing method: standard (argmax) or conf_threshold")
    parser.add_argument("--conf_threshold", default=0.70, type=float,
                        help="confidence threshold when using conf_threshold post-processing")
    parser.add_argument("--save_probs", action="store_true",
                        help="also save the foreground softmax probability map (<key>_prob.nii.gz, "
                             "float32) from the SAME TTA-averaged tensor used for argmax, so that "
                             "{prob>0.5} exactly reproduces the saved mask. Enables threshold and "
                             "calibration analysis.")

    # Distributed arguments
    parser.add_argument("--distributed", action="store_true", help="use distributed inference")
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--rank", default=0, type=int, help="rank of the process")
    parser.add_argument("--local_rank", default=0, type=int, help="local rank for distributed training")
    parser.add_argument("--dist_url", default="env://", type=str, help="url for distributed training")
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")

    return parser


def setup_distributed(args):
    """Initialize distributed environment."""
    if args.distributed:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            print("Distributed environment variables not set. Using single GPU.")
            args.distributed = False
            return
        
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        dist.barrier()
        
        if args.rank == 0:
            print(f"Distributed initialized: world_size={args.world_size}")


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    """Check if this is the main process."""
    return not args.distributed or args.rank == 0


def create_transforms(args):
    """Create preprocessing and post-processing transforms."""
    
    transform_list = [
        transforms.LoadImaged(keys=["image"]),
        channel_firstd(keys=["image"]),
    ]
    
    if not args.skip_orientation:
        transform_list.append(
            transforms.Orientationd(keys=["image"], axcodes="RAS")
        )
    
    transform_list.extend([
        transforms.Spacingd(
            keys=["image"], 
            pixdim=(args.space_x, args.space_y, args.space_z), 
            mode=("bilinear")
        ),
        make_intensity_transform(["image"], args),
        transforms.CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        transforms.SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, 0)),
        transforms.SpatialPadd(keys=["image"], spatial_size=(0, 0, args.roi_z), method='end'),
        transforms.EnsureTyped(keys=["image"], track_meta=True),
    ])
    
    test_transform = transforms.Compose(transform_list)
    return test_transform


def make_post_transform(test_transform, use_legacy_meta_keys):
    invert_kwargs = dict(
        keys="pred",
        transform=test_transform,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True,
        allow_missing_keys=False,
    )
    if use_legacy_meta_keys:
        invert_kwargs.update(
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
        )
    return transforms.Compose([
        transforms.EnsureTyped(keys="pred"),
        transforms.Invertd(**invert_kwargs),
    ])


def invert_prediction(batch_item, post_transform_modern, post_transform_legacy):
    """Invert predictions across MONAI metadata layouts."""
    prefer_legacy = isinstance(batch_item, dict) and "image_meta_dict" in batch_item
    ordered = (
        [post_transform_legacy, post_transform_modern]
        if prefer_legacy
        else [post_transform_modern, post_transform_legacy]
    )
    last_error = None
    for transform in ordered:
        try:
            return transform(batch_item)
        except Exception as error:
            last_error = error
    raise RuntimeError(f"Failed to invert prediction with available MONAI metadata layouts: {last_error}")


def _reconcile_ckpt_geometry(args, checkpoint, path=""):
    """
    Guard against eval/train config drift. Checkpoints store their training args (vars(args)); we compare
    the geometry/arch the caller is using against what the model was trained with. ARCH mismatches print a
    loud warning (strict state_dict load usually catches these anyway); ROI mismatches HARD-FAIL, because
    they load cleanly but silently corrupt results - this is exactly the roi_z=96-vs-64 trap. Override
    deliberately with --allow_roi_mismatch. Older checkpoints without stored args are skipped (warned once).
    """
    targs = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    name = os.path.basename(path) if path else "checkpoint"
    # Checkpoints in the wild store `args` two ways: as vars(args) (a plain dict,
    # what trainer.py writes now) or as the raw argparse.Namespace (older runs).
    # Normalise to a dict so subscripting below is always valid.
    if targs is not None and not isinstance(targs, dict):
        targs = getattr(targs, "__dict__", None)
    if not targs:
        print(f"[load_model][WARN] {name} has no stored training args - cannot verify geometry/arch.")
        return
    for k in ("model_name", "swin_use_v2", "n_decoder_channels", "resolution_factor",
              "head_upsample", "feature_size", "out_channels", "norm_name",
              "use_lora", "lora_rank", "lora_members", "decoder_ensemble", "lora_decoder_ensemble"):
        if k in targs and hasattr(args, k) and getattr(args, k) != targs[k]:
            print(f"[load_model][WARN] {k}: eval={getattr(args, k)} != train={targs[k]} ({name})")
    roi_bad = [k for k in ("roi_x", "roi_y", "roi_z")
               if k in targs and hasattr(args, k) and getattr(args, k) != targs[k]]
    if roi_bad and not getattr(args, "allow_roi_mismatch", False):
        diffs = ", ".join(f"{k}: eval={getattr(args, k)} train={targs[k]}" for k in roi_bad)
        raise ValueError(
            f"[load_model] inference ROI differs from checkpoint training args ({name}): {diffs}. "
            f"This loads cleanly but silently corrupts results (the z=96-vs-64 bug). Match the training "
            f"ROI, or pass --allow_roi_mismatch to override deliberately.")
    if targs:
        print(f"[load_model] geometry OK vs {name}: roi=({args.roi_x},{args.roi_y},{args.roi_z}) "
              f"model={getattr(args, 'model_name', '?')}")


def load_model(args, device):
    if args.model_name == 'smit':
        config = configs_smit.get_SMIT_128_bias_True_upernet() if args.use_upernet else configs_smit.get_SMIT_128_bias_True()
        model = smit.SMIT_3D_Seg(config,
                                 out_channels=args.out_channels,
                                 img_size=(args.roi_x, args.roi_y, args.roi_z),
                                 norm_name=args.norm_name)
    elif args.model_name == 'swinunetr':
        from models import swin_nvidia
        model = swin_nvidia.SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            norm_name=args.norm_name,
        )
    # 'swinv2' = canonical key for the MONAI SwinUNETR-V2 architecture.
    # 'voco_swinunetr' kept as a legacy alias only (the arch is SwinV2, not VoCo;
    # pretraining is chosen via the checkpoint, not this key).
    elif args.model_name in ('swinv2', 'voco_swinunetr'):
        model = build_monai_swin_unetr(args)
    elif args.model_name == 'effidec3d':
        model = build_effidec3d(args)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    # LoRA/LoRA-Ensemble: inject the SAME wrapping used in training BEFORE loading, or the checkpoint
    # keys (frozen base `w`, adapters `w_a/w_b`, and the EnsembleSeg `model.` prefix) won't match.
    if args.use_lora:
        from models.lora_ensemble import wrap_window_attention, freeze_for_lora, EnsembleSeg
        enc_attr = "transformer" if args.model_name == "smit" else "swinViT"
        def _clone():
            if args.model_name == "effidec3d":
                return build_effidec3d(args)
            if args.model_name in ("swinv2", "voco_swinunetr"):
                return build_monai_swin_unetr(args)
            raise ValueError(f"LoRA decoder ensembles unsupported for model_name={args.model_name}")
        if getattr(args, "lora_decoder_ensemble", False) and args.lora_members > 1:
            from models.lora_ensemble.decoder_ensemble import build_lora_decoder_ensemble
            model = build_lora_decoder_ensemble(model, _clone, args.lora_members, args.lora_rank,
                                                encoder_attr=enc_attr)
        elif getattr(args, "decoder_ensemble", False) and args.lora_members > 1:
            from models.lora_ensemble.decoder_ensemble import build_decoder_ensemble
            wrap_window_attention(getattr(model, enc_attr), rank=args.lora_rank, n_members=1, single=True)
            freeze_for_lora(model, encoder_attr=enc_attr)
            model = build_decoder_ensemble(model, _clone, args.lora_members, encoder_attr=enc_attr)
        else:
            single = args.lora_members <= 1
            wrap_window_attention(getattr(model, enc_attr), rank=args.lora_rank,
                                  n_members=args.lora_members, single=single)
            freeze_for_lora(model, encoder_attr=enc_attr)  # freeze irrelevant at eval; keeps structure identical
            if not single:
                model = EnsembleSeg(model, args.lora_members)  # -> forward returns [N, B, C, ...]

    checkpoint = torch.load(args.pretrained_model_path, map_location="cpu")
    _reconcile_ckpt_geometry(args, checkpoint, args.pretrained_model_path)
    model_dict = checkpoint.get("state_dict", checkpoint)

    # Handle DDP state dict
    new_state_dict = {}
    for k, v in model_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model.to(device)
    
    return model


def run_inference_with_tta(model, val_inputs, args):
    """Run inference with optional test-time augmentation."""
    # LoRA-Ensemble: model(x) -> [N,B,C,...]. For probability/calibration correctness,
    # blend probabilities across sliding windows and TTA views, then return log-probs so the
    # existing downstream softmax()/argmax()/save_probs path recovers the same mean probabilities.
    is_lora_ensemble = getattr(args, "use_lora", False) and getattr(args, "lora_members", 1) > 1
    if is_lora_ensemble:
        from models.lora_ensemble import ensemble_reduce
        predictor = lambda x: ensemble_reduce(model(x))
    else:
        predictor = model

    if not args.use_tta:
        pred = sliding_window_inference(
            inputs=val_inputs,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            sw_batch_size=args.sw_batch_size,
            predictor=predictor,
            overlap=args.infer_overlap,
            mode="gaussian",
            progress=False
        )
        return torch.log(pred.clamp_min(1e-8)) if is_lora_ensemble else pred

    preds = []
    for do_flip in [False, True]:
        aug_inputs = val_inputs

        # Apply flip
        if do_flip:
            aug_inputs = torch.flip(aug_inputs, dims=(2,))

        # Run inference
        pred = sliding_window_inference(
            inputs=aug_inputs,
            roi_size=(args.roi_x, args.roi_y, args.roi_z),
            sw_batch_size=args.sw_batch_size,
            predictor=predictor,
            overlap=args.infer_overlap,
            mode="gaussian",
            progress=False
        )
        
        # Invert flip
        if do_flip:
            pred = torch.flip(pred, dims=(2,))
        
        preds.append(pred)

    pred = torch.mean(torch.stack(preds), dim=0)
    return torch.log(pred.clamp_min(1e-8)) if is_lora_ensemble else pred

def apply_postprocessing(pred_t, args):
    """Apply post-processing to get final segmentation labels."""
    if args.postproc == 'standard':
        pred_labels = torch.argmax(pred_t, dim=0).to(torch.int64)
    else:
        probs = torch.softmax(pred_t, dim=0)
        conf, pred_labels = torch.max(probs, dim=0)
        pred_labels[conf < args.conf_threshold] = 0
        pred_labels = pred_labels.to(torch.int64)
    
    return pred_labels


def _first_item(value):
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def get_image_metadata(batch_or_item):
    """Return image metadata across legacy dict and MONAI MetaTensor layouts."""
    if isinstance(batch_or_item, dict) and "image_meta_dict" in batch_or_item:
        return batch_or_item["image_meta_dict"]

    image = batch_or_item.get("image") if isinstance(batch_or_item, dict) else None
    if image is not None and hasattr(image, "meta"):
        return image.meta

    pred = batch_or_item.get("pred") if isinstance(batch_or_item, dict) else None
    if pred is not None and hasattr(pred, "meta"):
        return pred.meta

    return {}


def get_image_filename(batch_or_item):
    meta = get_image_metadata(batch_or_item)
    filename = meta.get("filename_or_obj")
    if filename is not None:
        return str(_first_item(filename))

    image = batch_or_item.get("image") if isinstance(batch_or_item, dict) else None
    if image is not None and hasattr(image, "meta"):
        filename = image.meta.get("filename_or_obj")
        if filename is not None:
            return str(_first_item(filename))

    return "<unknown>"


def get_image_affine(batch_or_item):
    meta = get_image_metadata(batch_or_item)
    affine = meta.get("original_affine")
    if affine is None:
        affine = meta.get("affine")
    if affine is None:
        return np.eye(4)

    affine = _first_item(affine)
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
    return np.asarray(affine)


def attach_pred_metadata(pred, image):
    meta = getattr(image, "meta", None)
    applied_operations = getattr(image, "applied_operations", None)
    return MetaTensor(pred, meta=meta, applied_operations=applied_operations)


def process_dataset(dataset_name, args, model, test_transform, post_transform_modern, post_transform_legacy, device):
    """Process a single dataset."""
    if is_main_process(args):
        print(f'Working on: {dataset_name}')
    
    output_directory = os.path.join(
        args.results_dir,
        args.output_dir,
        args.pred_subdir if args.pred_subdir else args.model_name,
        dataset_name
    )
    
    if is_main_process(args):
        os.makedirs(output_directory, exist_ok=True)
    
    if args.distributed:
        dist.barrier()
    
    datalist_json = os.path.join(args.data_dir, args.json_list)
    test_files = load_decathlon_datalist(
        datalist_json, 
        True, 
        dataset_name, 
        base_dir=args.data_dir
    )
    
    test_ds = data.Dataset(test_files, transform=test_transform)
    
    # Use distributed sampler if distributed
    if args.distributed:
        sampler = DistributedSampler(test_ds, shuffle=False)
    else:
        sampler = None
    
    test_loader = data.DataLoader(
        test_ds, 
        batch_size=1, 
        shuffle=False,
        sampler=sampler,
        pin_memory=True
    )
    
    # Progress bar only on main process
    if is_main_process(args):
        loader_iter = tqdm(test_loader, desc=f"{dataset_name}")
    else:
        loader_iter = test_loader
    
    with torch.no_grad():
        for i, batch in enumerate(loader_iter):
            try:
                val_inputs = batch["image"].to(device)
                
                img_name = make_seg_filename(get_image_filename(batch))
                
                batch["pred"] = attach_pred_metadata(
                    run_inference_with_tta(model, val_inputs, args),
                    batch["image"],
                )
                
                batch = [invert_prediction(i, post_transform_modern, post_transform_legacy) for i in decollate_batch(batch)]
                
                b0 = batch[0]
                pred_t = b0['pred']
                pred_labels = apply_postprocessing(pred_t, args)
                
                seg_np = pred_labels.cpu().numpy().astype(np.uint8)
                affine = get_image_affine(b0)
                
                output_path = os.path.join(output_directory, img_name)
                nib.save(nib.Nifti1Image(seg_np, affine), output_path)

                # Optional: foreground probability map for Gate-A / threshold analysis.
                # softmax of the SAME pred_t that argmax produced the mask from, in the SAME
                # (native) space with the SAME affine -> {prob > 0.5} reproduces seg_np exactly
                # (for 2-class argmax). So false negatives (GT=1, prob<0.5) are self-consistent.
                if getattr(args, "save_probs", False):
                    # NIfTI-1 has no float16 datatype -> save float32 (prob in [0,1]).
                    fg_prob = torch.softmax(pred_t.float(), dim=0)[1].cpu().numpy().astype(np.float32)
                    prob_path = os.path.join(output_directory, img_name.replace("_seg.nii.gz", "_prob.nii.gz"))
                    nib.save(nib.Nifti1Image(fg_prob, affine), prob_path)

            except Exception as e:
                print(f"\n[Rank {args.rank}] Error processing {get_image_filename(batch)}: {e}")
                continue
    
    if args.distributed:
        dist.barrier()
    
    if is_main_process(args):
        print(f"Completed: {dataset_name}\n")


def main():
    parser = setup_argparser()
    args = parser.parse_args()
    if args.decoder_ensemble and args.lora_decoder_ensemble:
        raise ValueError("--decoder_ensemble and --lora_decoder_ensemble are mutually exclusive.")
    if args.decoder_ensemble:
        if not args.use_lora:
            raise ValueError("--decoder_ensemble requires --use_lora.")
        if args.lora_members <= 1:
            raise ValueError("--decoder_ensemble requires --lora_members > 1.")
    if args.lora_decoder_ensemble:
        if not args.use_lora:
            raise ValueError("--lora_decoder_ensemble requires --use_lora.")
        if args.lora_members <= 1:
            raise ValueError("--lora_decoder_ensemble requires --lora_members > 1.")
    
    # Setup distributed
    setup_distributed(args)
    
    # Set device
    if args.distributed:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_main_process(args):
        print("=" * 60)
        print("Inference Configuration")
        print("=" * 60)
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
        print(f"\nDevice: {device}")
        print(f"Distributed: {args.distributed}")
        if args.distributed:
            print(f"World size: {args.world_size}")
        print(f"Orientation transform: {'SKIPPED' if args.skip_orientation else 'RAS'}")
        print(f"TTA: {args.use_tta}")
        print("=" * 60 + "\n")
    
    test_transform = create_transforms(args)
    post_transform_modern = make_post_transform(test_transform, use_legacy_meta_keys=False)
    post_transform_legacy = make_post_transform(test_transform, use_legacy_meta_keys=True)
    
    if is_main_process(args):
        print("Loading model...")
    
    model = load_model(args, device)
    
    if is_main_process(args):
        print(f"Model: {args.model_name}")
        print(f"Output channels: {args.out_channels}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Post-processing: {args.postproc}")
        if args.postproc == 'conf_threshold':
            print(f"Confidence threshold: {args.conf_threshold}")
        print()
    
    for dataset_name in args.datasets:
        process_dataset(
            dataset_name, 
            args, 
            model, 
            test_transform, 
            post_transform_modern,
            post_transform_legacy,
            device
        )
    
    cleanup_distributed()
    
    if is_main_process(args):
        print("Inference completed successfully!")


if __name__ == "__main__":
    main()
