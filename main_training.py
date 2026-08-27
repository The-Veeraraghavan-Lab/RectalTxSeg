# Copyright 2021 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR, make_warmup_cosine_then_cyclic
from trainer import run_training

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

from models import smit, configs_smit
from models import swin_nvidia
from models.swin_voco_utils import build_monai_swin_unetr, load_swin_encoder_pretrained, build_effidec3d

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument("--logdir", default="testrun_tv1", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="data_rectal", type=str, help="dataset directory")
parser.add_argument("--json_list", default="Trainval_set1.json", type=str, help="dataset json file")
parser.add_argument("--train_key", default="training", type=str, help="JSON key for the training split (e.g. training, training_c0, training_c1)")
parser.add_argument("--val_key", default="validation", type=str, help="JSON key for the validation split")
parser.add_argument("--label_key", default="label", type=str,
                    help="primary label field name in the dataset json")
parser.add_argument("--fallback_label_key", default="label_plus", type=str,
                    help="alternate label field name used when present in the dataset json")
parser.add_argument("--prefer_label_plus", action="store_true",
                    help="rewrite /label/ paths to /label_plus/ when loading labels")
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")

parser.add_argument("--max_epochs", default=500, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--train_num_samples", default=4, type=int,
                    help="random crops sampled per training volume by RandCropByPosNegLabeld")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")

parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--optim_lr", default=3e-4, type=float, help="optimization learning rate")
parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
# --- LoRA / LoRA-Ensemble (arXiv:2405.14438): freeze the PRE-TRAINED encoder, train low-rank attention adapters. ---
parser.add_argument("--use_lora", action="store_true",
                    help="freeze pretrained encoder; train low-rank adapters on attention qkv/proj")
parser.add_argument("--lora_rank", default=4, type=int, help="LoRA rank (paper sweet spot 4-8)")
parser.add_argument("--lora_members", default=1, type=int,
                    help="1 = plain LoRA (parameter efficiency); >1 = LoRA-Ensemble (implicit ensemble, uncertainty)")
parser.add_argument("--allow_random_lora", action="store_true",
                    help="permit --use_lora WITHOUT --use_ssl_pretrained (LoRA around a RANDOM frozen encoder; almost never intended)")
parser.add_argument("--decoder_ensemble", action="store_true",
                    help="LoRA + decoder-ensemble upper-bound: shared frozen encoder/adapters + N independent decoders; needs --lora_members>1")
parser.add_argument("--lora_decoder_ensemble", action="store_true",
                    help="paper-closest LoRA-Ensemble segmentation port: N independent LoRA adapters + N independent decoders")
parser.add_argument("--mock_val", action="store_true",
                    help="run ONE validation pass at start (sanity check only, not saved) before normal training")
parser.add_argument("--use_multiscale_aug", action="store_true",
                    help="pyramid-lite: in-plane multi-scale (RandZoom, z fixed) so the encoder is scale-robust to small tumours")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")

parser.add_argument("--model_name", default = 'smit', type =str, help="model name (configuration)")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--swin_depths", default=(2, 2, 2, 2), type=int, nargs=4,
                    help="MONAI SwinUNETR stage depths for swinv2 (SwinV2 SwinUNETR)")
parser.add_argument("--swin_num_heads", default=(3, 6, 12, 24), type=int, nargs=4,
                    help="MONAI SwinUNETR attention heads for swinv2 (SwinV2 SwinUNETR)")
parser.add_argument("--swin_use_v2", action="store_true",
                    help="Use MONAI SwinUNETR V2 blocks for swinv2 (SwinV2 SwinUNETR)")
# --- effidec3d (EffiDec3D decoder on the SwinV2 encoder) ---
parser.add_argument("--n_decoder_channels", default=48, type=int,
                    help="effidec3d: fixed channel width across all decoder stages")
parser.add_argument("--resolution_factor", default=2, type=int,
                    help="effidec3d: coarsest output stride (1=full res, 2=half res, ...)")
parser.add_argument("--head_upsample", default="trilinear", type=str,
                    choices=["none", "trilinear", "upconv", "upconv_refine"],
                    help="effidec3d: how the reduced-res output is restored to full res")

parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--dataset_backend", default="cache", choices=["normal", "cache", "persistent"],
                    help="training/validation dataset backend")
parser.add_argument("--persistent_cache_root", default=None, type=str,
                    help="root directory for MONAI PersistentDataset caches")
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
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")

parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")

parser.add_argument("--lambda_dice", default=1.0, type=float, help="value for dice loss in joint function")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")

parser.add_argument("--use_ssl_pretrained", action="store_true",
                    help="Load self-supervised pretrained backbone weights before supervised training.")  
parser.add_argument("--ssl_weights_path", type=str, default="model_smit_ct10k.pth",
                    help="Path to the SSL checkpoint (e.g., SMIT CT10K). Assumes in *folder pretrained_models")
parser.add_argument("--use_adv_loader", action="store_true", help="Use advanced MRI data loader")
parser.add_argument("--exclude_last_ds", action="store_true", help="Use model without last downsample in transformer")
parser.add_argument("--use_tumor_aug", action="store_true", help="Use tumor-specific intensity augmentation")
parser.add_argument("--tumor_label", default=1, type=int, 
                    help="label index of tumor for RandTumorIntensityd")
parser.add_argument("--use_upernet", action="store_true", help="Use UPERNET decoder")

def main():
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
    print(args)
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    if args.use_adv_loader:
        print("=====using advanced data loader=====")
        from utils.data_utils import get_loader_v2_mri_adv
        loader = get_loader_v2_mri_adv(args)
    else:
        print("=====using standard data loader=====")
        from utils.data_utils import get_loader_v2_mri
        loader = get_loader_v2_mri(args)
        
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    def get_patch_embed_weight(model, model_name):
        if model_name == 'smit':
            return model.transformer.patch_embed.norm.weight
        elif model_name == 'swinunetr':
            return model.swinViT.patch_embed.proj.weight
        elif model_name in ('swinv2', 'voco_swinunetr', 'effidec3d'):
            return model.swinViT.patch_embed.proj.weight

    if args.use_upernet:
        config = configs_smit.get_SMIT_128_bias_True_upernet()
    else:
        config = configs_smit.get_SMIT_128_bias_True()
    
    if args.model_name == 'smit':
        model = smit.SMIT_3D_Seg(config,
                                 out_channels = args.out_channels,
                                 img_size = (args.roi_x, args.roi_y, args.roi_z),
                                 norm_name = args.norm_name)
    elif args.model_name == 'swinunetr':
        model = swin_nvidia.SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=args.feature_size,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=False,
            norm_name=args.norm_name,
        )
    # 'swinv2' is the canonical key for the MONAI SwinUNETR-V2 architecture.
    # 'voco_swinunetr' is kept only as a legacy alias: VoCo was merely the first
    # PRETRAINING wired through this path, but the network is SwinV2, not VoCo-specific.
    # Pretraining (VoCo / VoxelFox / ...) is selected via --ssl_weights_path, not this key.
    elif args.model_name in ('swinv2', 'voco_swinunetr'):
        model = build_monai_swin_unetr(args)
    # EffiDec3D: same SwinV2 encoder, channel-reduced / reduced-resolution decoder.
    elif args.model_name == 'effidec3d':
        model = build_effidec3d(args)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    print(get_patch_embed_weight(model, args.model_name))

    if args.use_ssl_pretrained:
        try:
            if args.model_name == 'smit':
                model_dict = torch.load(os.path.join('pretrained_models', args.ssl_weights_path), map_location='cpu')
                
                pretrained_dict = model_dict['student']
                for key in list(pretrained_dict.keys()):
                    pretrained_dict[key.replace('module.backbone.', '')] = pretrained_dict.pop(key)
                model.load_state_dict(pretrained_dict, strict=False)
                print("Using pretrained self-supervised SMIT backbone weights !")

            elif args.model_name == 'swinunetr':
                model_dict = torch.load(os.path.join('pretrained_models', args.ssl_weights_path), map_location='cpu')

                # Normalise key namespace: ct10k weights nest backbone under 'module.swinViT.'
                # but load_from() expects 'module.' directly - strip the extra prefix if present.
                sd = model_dict['state_dict']
                # Remap ct10k keys to match load_from() expectations:
                #   module.swinViT.* -> module.*
                #   mlp.linear1.*/mlp.linear2.* -> mlp.fc1.*/mlp.fc2.*
                remapped = {}
                for k, v in sd.items():
                    k = k.replace('module.swinViT.', 'module.')
                    k = k.replace('mlp.linear1.', 'mlp.fc1.')
                    k = k.replace('mlp.linear2.', 'mlp.fc2.')
                    remapped[k] = v
                model_dict['state_dict'] = remapped

                model.load_from(model_dict)
                print(f"Loaded SwinUNETR pretrained weights from {args.ssl_weights_path} via load_from()!")        
            elif args.model_name in ('swinv2', 'voco_swinunetr', 'effidec3d'):
                # effidec3d shares the SwinV2 encoder (model.swinViT); the decoder stays random.
                load_swin_encoder_pretrained(model, args.ssl_weights_path, strict=False)
                print(f"Loaded SwinV2 SwinUNETR encoder weights from {args.ssl_weights_path}!")
            
        except Exception as e:
            raise ValueError(f"Failed to load pretrained weights: {e}")
    
    print(get_patch_embed_weight(model, args.model_name))

    # --- LoRA / LoRA-Ensemble: freeze the pre-trained encoder, train low-rank attention adapters ---
    # Must run AFTER the SSL encoder load (encoder keeps its pretrained weights) and BEFORE the optimizer,
    # DDP wrap, and val_predictor below.
    if args.use_lora:
        if not args.use_ssl_pretrained and not args.allow_random_lora:
            raise ValueError(
                "--use_lora expects a PRE-TRAINED encoder (--use_ssl_pretrained): LoRA adapts a frozen "
                "backbone, so freezing a RANDOM encoder and training tiny adapters around it is almost "
                "never intended. Pass --allow_random_lora to override deliberately.")
        from models.lora_ensemble import (wrap_window_attention, freeze_for_lora, count_params,
                                          EnsembleSeg, ensemble_reduce)
        enc_attr = "transformer" if args.model_name == "smit" else "swinViT"
        def _clone():
            if args.model_name == "effidec3d":
                return build_effidec3d(args)
            if args.model_name in ("swinv2", "voco_swinunetr"):
                return build_monai_swin_unetr(args)
            raise ValueError(f"LoRA decoder ensembles unsupported for model_name={args.model_name}")
        if getattr(args, "lora_decoder_ensemble", False) and args.lora_members > 1:
            # Paper-closest segmentation port: each member gets its own LoRA adapters and its own decoder/head.
            # Every member starts from the same SSL encoder weights; only the frozen base weights are shared in
            # initialization, not as runtime parameter objects.
            from models.lora_ensemble.decoder_ensemble import build_lora_decoder_ensemble, count_params_dedup
            model = build_lora_decoder_ensemble(model, _clone, args.lora_members, args.lora_rank,
                                                encoder_attr=enc_attr)
            cp = count_params_dedup(model)
            print(f"[LoRA] LORA-DECODER-ENSEMBLE x{args.lora_members} rank={args.lora_rank} | "
                  f"per-member adapters + decoders | trainable={cp['trainable']:,}/{cp['total']:,} ({cp['pct']:.2f}%)")
        elif getattr(args, "decoder_ensemble", False) and args.lora_members > 1:
            # Upper-bound variant: SHARED frozen encoder + SHARED LoRA adapters, N INDEPENDENT decoders (diversity in
            # the decoders, where the mask is formed). Encoder is one shared module -> counted once, so total
            # stays below the full-VoxelFox baseline even for N decoders.
            from models.lora_ensemble.decoder_ensemble import build_decoder_ensemble, count_params_dedup
            wrap_window_attention(getattr(model, enc_attr), rank=args.lora_rank, n_members=1, single=True)
            freeze_for_lora(model, encoder_attr=enc_attr)
            model = build_decoder_ensemble(model, _clone, args.lora_members, encoder_attr=enc_attr)
            cp = count_params_dedup(model)
            print(f"[LoRA] DECODER-ENSEMBLE x{args.lora_members} rank={args.lora_rank} | shared encoder + "
                  f"{args.lora_members} decoders | trainable={cp['trainable']:,}/{cp['total']:,} ({cp['pct']:.2f}%)")
        else:
            single = args.lora_members <= 1
            n_adapted = wrap_window_attention(getattr(model, enc_attr), rank=args.lora_rank,
                                              n_members=args.lora_members, single=single)
            freeze_for_lora(model, encoder_attr=enc_attr)
            if not single:
                model = EnsembleSeg(model, args.lora_members)   # forward(x) -> [N, B, C, D, H, W]
            cp = count_params(model)
            print(f"[LoRA] mode={'single' if single else f'ensemble x{args.lora_members}'} "
                  f"rank={args.lora_rank} adapted={n_adapted} attn modules | "
                  f"trainable={cp['trainable']:,}/{cp['total']:,} ({cp['pct']:.2f}%)")

    os.makedirs(args.logdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.logdir, 'random_init_weights.pth'))
    
    dice_loss = DiceCELoss(
        include_background=True, to_onehot_y=True, softmax=True,
        squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr,
        lambda_dice=args.lambda_dice
        )

    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc = DiceMetric(include_background=False, reduction=MetricReduction.MEAN, get_not_nans=True)
    if args.use_lora and args.lora_members > 1:
        # ensemble: model(x) -> [N,B,C,...]; reduce to mean class-probabilities [B,C,...] for sliding window + Dice
        from models.lora_ensemble import ensemble_reduce
        val_predictor = lambda x: ensemble_reduce(model(x))
    else:
        val_predictor = model
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=val_predictor,
        overlap=args.infer_overlap,
    )
        
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    start_epoch = 0

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        # decoder_ensemble shares the encoder across N member forwards (shared LoRA params used multiple
        # times per step) -> DDP needs static_graph to avoid "mark a variable ready only once".
        use_static_graph = bool(getattr(args, "decoder_ensemble", False))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          find_unused_parameters=not use_static_graph,
                                                          static_graph=use_static_graph)
    # Only optimise trainable params (LoRA freezes the encoder; harmless for the base path where all are trainable).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params, lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
            )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = None

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
