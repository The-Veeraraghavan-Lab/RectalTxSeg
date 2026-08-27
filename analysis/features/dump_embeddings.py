"""Phase 1a — dump penultimate backbone features at lesion vs background voxels.

The linchpin probe's data collection. For each GT lesion we crop a fixed ROI, run the base effidec model,
and capture the features feeding the 1x1x1 SegHead (model.out) — i.e. the representation the final decision
is read off. We record, per sampled voxel: the feature vector, whether it is lesion or background, the
lesion's volume/size-bin, whether the lesion was DETECTED (full-res ≥10% overlap), and the model's
foreground probability there.

This one dump answers two questions downstream (analysis/features/separability_probe.py, LOCAL):
  1. Do features SEPARATE missed <5cc lesions from background? (linear probe)  -> is the signal in the
     features or absent (representational failure)?
  2. Same features feed the radiomics-vs-deep-features comparison (Phase 3).

With head_upsample='trilinear' (our config) model.out runs on the /resolution_factor decoder features, so
the hooked tensor is at reduced res; the GT is aligned to it with adaptive max-pool (a reduced voxel is
lesion if ANY full-res lesion voxel maps into it — preserves small tumours).

  python analysis/features/dump_embeddings.py \
    --base_ckpt runs/rectal_effidec3d_96x96x64_pretrained/model_final.pt \
    --data_dir data_rectal --json_list Trainval_set1.json --mode testing \
    --out_npz analysis/features/embeddings_effidec_testing.npz
"""
import os
import os.path as osp
import sys
import json
import argparse
from types import SimpleNamespace

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from monai import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.swin_voco_utils import build_effidec3d, build_monai_swin_unetr
from utils.lesion_utils import _gt_lesions, bin_label


def build_base(args):
    if args.model_name == "effidec3d":
        return build_effidec3d(args)
    if args.model_name in ("swinv2", "voco_swinunetr"):
        return build_monai_swin_unetr(args)
    raise ValueError(f"unsupported base architecture: {args.model_name}")


def load_base_ckpt(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded checkpoint {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")


def effidec_args(cli):
    """Arch namespace matching the base effidec run (train.sh effidec3d)."""
    return SimpleNamespace(
        model_name="effidec3d", in_channels=1, out_channels=2, feature_size=48,
        swin_depths=(2, 2, 2, 2), swin_num_heads=(3, 6, 12, 24), swin_use_v2=True,
        n_decoder_channels=48, resolution_factor=2, head_upsample="trilinear",
        norm_name="instance", dropout_rate=0.0, dropout_path_rate=0.0, use_checkpoint=False,
        spatial_dims=3, roi_x=cli.roi_x, roi_y=cli.roi_y, roi_z=cli.roi_z,
    )


def build_transform(cli):
    """Image+label preprocessing matching the base val pipeline (RAS, 1mm iso, [0,800]->[0,1])."""
    return transforms.Compose([
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.EnsureChannelFirstd(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        transforms.ScaleIntensityRanged(keys=["image"], a_min=cli.a_min, a_max=cli.a_max,
                                        b_min=0.0, b_max=1.0, clip=True),
        transforms.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        transforms.ToTensord(keys=["image", "label"]),
    ])


def roi_crop(img, lab, center, roi):
    """Deterministic ROI crop centred on `center`, clamped to fit, end-padded if the volume is smaller.
    img: (1,1,*V) tensor, lab: (1,1,*V) tensor -> (1,1,*roi), (1,1,*roi)."""
    V = img.shape[2:]
    lo = [max(0, min(int(center[i]) - roi[i] // 2, max(0, V[i] - roi[i]))) for i in range(3)]
    ic = img[:, :, lo[0]:lo[0]+roi[0], lo[1]:lo[1]+roi[1], lo[2]:lo[2]+roi[2]]
    lc = lab[:, :, lo[0]:lo[0]+roi[0], lo[1]:lo[1]+roi[1], lo[2]:lo[2]+roi[2]]
    pad = (0, max(0, roi[2]-ic.shape[4]), 0, max(0, roi[1]-ic.shape[3]), 0, max(0, roi[0]-ic.shape[2]))
    return F.pad(ic, pad), F.pad(lc, pad)


def gt_path(data_dir, item):
    if item.get("label"):
        p = osp.join(data_dir, item["label"]); return p if osp.exists(p) else None
    if "imagesTs/" in item.get("image", ""):
        p = osp.join(data_dir, item["image"]).replace("imagesTs/", "labelsTs/").replace("_z800", "")
        return p if osp.exists(p) else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--data_dir", default="data_rectal"); ap.add_argument("--json_list", default="Trainval_set1.json")
    ap.add_argument("--mode", default="testing")
    ap.add_argument("--roi_x", type=int, default=96); ap.add_argument("--roi_y", type=int, default=96); ap.add_argument("--roi_z", type=int, default=64)
    ap.add_argument("--a_min", type=float, default=0.0); ap.add_argument("--a_max", type=float, default=800.0)
    ap.add_argument("--n_pos", type=int, default=200, help="max lesion voxels sampled per lesion")
    ap.add_argument("--n_neg", type=int, default=200, help="background voxels sampled per lesion")
    ap.add_argument("--det_thresh", type=float, default=0.10, help="lesion detected if >=this fraction overlaps pred fg")
    ap.add_argument("--out_npz", default="analysis/features/embeddings_effidec_testing.npz")
    cli = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = effidec_args(cli)
    model = build_base(a)
    load_base_ckpt(model, cli.base_ckpt)
    model = model.to(device).eval()

    feat_store = {}
    def hook(_m, inp, out):
        feat_store["f"] = inp[0].detach()      # (1, Cf, *reduced) — features feeding the SegHead
        feat_store["z"] = out.detach()         # (1, 2, *reduced)  — reduced-res logits
    h = model.out.register_forward_hook(hook)

    tf = build_transform(cli)
    items = json.load(open(osp.join(cli.data_dir, cli.json_list)))[cli.mode]
    roi = (cli.roi_x, cli.roi_y, cli.roi_z)
    rng = np.random.default_rng(0)

    X, y, prob, vol, det, cid = [], [], [], [], [], []
    n_cases = n_les = 0
    with torch.no_grad():
        for it in items:
            gp = gt_path(cli.data_dir, it)
            if gp is None:
                continue
            d = tf({"image": osp.join(cli.data_dir, it["image"]), "label": gp})
            img, lab = d["image"].unsqueeze(0).float(), d["label"].unsqueeze(0).float()   # (1,1,*V)
            sp = (1.0, 1.0, 1.0)
            glab, gvols = _gt_lesions((lab[0, 0].cpu().numpy() == 1), sp)
            if len(gvols) == 0:
                continue
            n_cases += 1
            key = osp.basename(it["image"]).split(".")[0]
            for c, vcc in enumerate(gvols, 1):
                # centroid of this lesion (full res)
                idx = np.argwhere(glab == c)
                ctr = idx.mean(0)
                les_full = torch.from_numpy((glab == c).astype(np.float32))[None, None]  # (1,1,*V)
                ic, lc = roi_crop(img, les_full, ctr, roi)                                # (1,1,*roi)
                logits_full = model(ic.to(device))                                        # full-res logits
                f = feat_store["f"]                                                       # (1,Cf,*red)
                red = f.shape[2:]
                # lesion mask aligned to feature grid: adaptive max-pool keeps small lesions
                lred = F.adaptive_max_pool3d(lc.to(device), output_size=red)[0, 0]        # (*red)
                fgp_red = torch.softmax(feat_store["z"], dim=1)[0, 1]                      # (*red) reduced fg prob
                fv = f[0].permute(1, 2, 3, 0).reshape(-1, f.shape[1]).cpu().numpy()       # (Nred, Cf)
                lflat = lred.reshape(-1).cpu().numpy() > 0.5
                pflat = fgp_red.reshape(-1).cpu().numpy()
                # lesion-level detection at FULL res
                pred_fg_full = (logits_full.argmax(1)[0] == 1).cpu().numpy()
                les_roi_full = (lc[0, 0].cpu().numpy() > 0.5)
                tot = int(les_roi_full.sum())
                detected = int((pred_fg_full & les_roi_full).sum() >= cli.det_thresh * max(tot, 1))
                sb = bin_label(vcc)
                pos_i = np.where(lflat)[0]; neg_i = np.where(~lflat)[0]
                if len(pos_i) == 0:
                    continue
                if len(pos_i) > cli.n_pos:
                    pos_i = rng.choice(pos_i, cli.n_pos, replace=False)
                if len(neg_i) > cli.n_neg:
                    neg_i = rng.choice(neg_i, cli.n_neg, replace=False)
                for grp, yy in ((pos_i, 1), (neg_i, 0)):
                    for i in grp:
                        X.append(fv[i]); y.append(yy); prob.append(float(pflat[i]))
                        vol.append(float(vcc)); det.append(detected); cid.append(key + f"_les{c}_{sb}")
                n_les += 1
            if n_cases % 40 == 0:
                print(f"  {n_cases} cases, {n_les} lesions...", flush=True)
    h.remove()

    os.makedirs(osp.dirname(cli.out_npz), exist_ok=True)
    np.savez_compressed(cli.out_npz,
                        X=np.asarray(X, np.float32), y=np.asarray(y, np.int8),
                        prob=np.asarray(prob, np.float32), vol_cc=np.asarray(vol, np.float32),
                        detected=np.asarray(det, np.int8), lesion_id=np.asarray(cid),
                        feat_dim=np.int32(np.asarray(X).shape[1] if X else 0))
    print(f"\nsaved {len(y)} voxel samples from {n_les} lesions ({n_cases} cases) -> {cli.out_npz}")
    print(f"  feat_dim={np.asarray(X).shape[1] if X else 0}; "
          f"pos={int(np.sum(y))} neg={len(y)-int(np.sum(y))}")
    print("NEXT (local): python analysis/features/separability_probe.py --npz " + cli.out_npz)


if __name__ == "__main__":
    main()
