"""Tumor-anchored DEEP features (the "backdrop" arm, parallels the CERR radiomics arm).

Recipe (tuned for rectal, kept deliberately simple/defensible):
  * one 64^3 crop centered on the tumor-cluster centroid, at 1.0mm iso - the SAME resolution the seg model
    was trained/run at, so features stay in-distribution (64mm FOV fully contains 96% of tumors; the rest
    keep their core+margin). Multi-crop is a trivial later extension (jitter the center).
  * run the EffiDec3D SwinV2 encoder (model.swinViT) and take the "set of 4" = the 4 encoder levels
    [x0..x3]; global-average-pool each and concat -> one per-case deep vector. --pool masked restricts the
    pool to inside the contour (the stricter, delineation-sensitive variant).
  * GT run anchors on the GT mask; AI run anchors on the LARGEST connected component of the prediction
    (GT-free) so crops never center on FP islands -- identical discipline to the radiomics arm.

One contour per run -> one CSV of plain feat_* columns keyed by id (same schema as radiomics_*.csv).

Run on the server (has torch + the run checkpoints):
  # GT reference
  python analysis/features/extract_deep_features.py --contour gt \
      --base_ckpt runs/rectal_effidec3d_96x96x64_pretrained/model_final.pt \
      --results_dir rectal_effidec3d_96x96x64_pretrained --modelname effidec3d \
      --out analysis/features/deep_gt.csv
  # VoxelFox AI contour
  python analysis/features/extract_deep_features.py --contour ai \
      --base_ckpt runs/rectal_effidec3d_96x96x64_pretrained/model_final.pt \
      --results_dir rectal_effidec3d_96x96x64_pretrained --modelname effidec3d \
      --out analysis/features/deep_voxelfox.csv
  # VoCo AI contour
  python analysis/features/extract_deep_features.py --contour ai \
      --base_ckpt runs/rectal_effidec3d_voco_96x96x64_pretrained/model_final.pt \
      --results_dir rectal_effidec3d_voco_96x96x64_pretrained --modelname effidec3d_voco \
      --out analysis/features/deep_voco.csv
  # SMIT (BSPC baseline-best) - different backbone, --arch smit:
  python analysis/features/extract_deep_features.py --contour ai --arch smit \
      --base_ckpt runs/rectal_smit_128x128x64_pretrained/model_final.pt \
      --results_dir rectal_smit_128x128x64_pretrained --modelname smit \
      --out analysis/features/deep_smit_ai.csv

NOTE deep features are CHECKPOINT-specific (unlike radiomics, whose GT csv is shared). VoxelFox / VoCo /
SMIT are different weights -> different feature spaces, so the GT descriptor is NOT shared. Each model is
its own extractor and needs BOTH a gt run and an ai run through the SAME --base_ckpt:
  per model M: --contour gt (-> deep_M_gt.csv) and --contour ai (-> deep_M_ai.csv), same base_ckpt.
The within-M comparison (AI-contour vs GT-contour, both through M's features) is the deep analogue of the
radiomics GT-vs-AI test. -> 6 deep runs total (gt+ai x voxelfox/voco/smit).
"""
import os, os.path as osp, sys, json, argparse, tempfile
from types import SimpleNamespace
import numpy as np, nibabel as nib, torch
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))  # repo root
from utils.lesion_utils import _key_name_part
from analysis.features.dump_embeddings import effidec_args, build_base, build_transform, gt_path, load_base_ckpt, roi_crop


def _lcc_pred_path(seg_path, min_island_cc=0.0):
    """GT-FREE cluster isolation: largest 26-connected component of a prediction -> temp nii path."""
    from scipy.ndimage import label, generate_binary_structure
    nii = nib.load(seg_path); m = nii.get_fdata() > 0.5
    if m.sum() == 0:
        return None
    lab, n = label(m, structure=generate_binary_structure(3, 3))
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    if min_island_cc > 0:
        vcc = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0
        sizes[sizes * vcc < min_island_cc] = 0
        if sizes.max() == 0:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
    out = (lab == int(sizes.argmax())).astype(np.uint8)
    tmp = tempfile.NamedTemporaryFile(suffix="_lcc.nii.gz", delete=False).name
    nib.save(nib.Nifti1Image(out, nii.affine, nii.header), tmp)
    return tmp


def _safe_unlink(path):
    if path and path.startswith(tempfile.gettempdir()) and osp.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


ENCODER_PREFIX = {"effidec3d": "swinViT.", "swinv2": "swinViT.", "smit": "transformer."}
DEFAULT_LEVELS = {"effidec3d": [1, 2, 3, 4], "swinv2": [1, 2, 3, 4], "smit": [0, 1, 2, 3]}


def _build_model(cli):
    """Return (model, encoder_features_fn). encoder_features_fn(crop_tensor) -> list of level tensors."""
    if cli.arch in ("effidec3d", "swinv2"):
        a = effidec_args(SimpleNamespace(roi_x=cli.roi, roi_y=cli.roi, roi_z=cli.roi)); a.model_name = cli.arch
        model = build_base(a)
        normalize = getattr(model, "normalize", True)
        return model, (lambda ic: model.swinViT(ic, normalize))          # -> [x0..x4]
    if cli.arch == "smit":
        from models import smit as smit_mod, configs_smit
        config = configs_smit.get_SMIT_128_bias_True()
        model = smit_mod.SMIT_3D_Seg(config, out_channels=2,
                                     img_size=(cli.roi, cli.roi, cli.roi), norm_name="instance")
        return model, (lambda ic: model.transformer(ic)[1])              # -> [stage0..stage3]
    raise ValueError(cli.arch)


def _assert_encoder_loaded(model, ckpt_path, arch):
    """Fail LOUD if the encoder didn't actually load - strict=False would otherwise yield garbage features.
    Guards arch/config mismatch across VoxelFox / VoCo / SMIT checkpoints."""
    prefix = ENCODER_PREFIX[arch]
    ck = torch.load(ckpt_path, map_location="cpu"); sd = ck.get("state_dict", ck)
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    msd = model.state_dict()
    enc = [k for k in msd if k.startswith(prefix)]
    ok = [k for k in enc if k in sd and tuple(sd[k].shape) == tuple(msd[k].shape)]
    frac = len(ok) / max(len(enc), 1)
    print(f"  encoder({prefix.rstrip('.')}) coverage: {len(ok)}/{len(enc)} = {frac:.1%}  [arch={arch}]")
    if frac < 0.9:
        raise RuntimeError(
            f"encoder '{prefix}' weights largely unmatched ({frac:.0%}) for {ckpt_path} with --arch {arch}. "
            "Wrong architecture/config for this checkpoint - features would be meaningless.")


def _pool_levels(levels, labcrop, pool):
    """levels: list of (1,C,*red). Return concat descriptor. gap=whole-crop average; masked=inside contour."""
    vecs = []
    for f in levels:
        if pool == "masked":
            red = f.shape[2:]
            mred = F.adaptive_max_pool3d(labcrop, output_size=red)[0, 0] > 0.5   # keep small lesions
            if mred.any():
                vecs.append(f[0][:, mred].mean(dim=1))
                continue
        vecs.append(f.mean(dim=(2, 3, 4))[0])                                    # GAP fallback / default
    return torch.cat(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", required=True)
    ap.add_argument("--data_dir", default="data_rectal"); ap.add_argument("--json_list", default="Trainval_set1.json")
    ap.add_argument("--mode", default="testing")
    ap.add_argument("--results_dir", required=True); ap.add_argument("--modelname", required=True)
    ap.add_argument("--contour", choices=["gt", "ai"], required=True)
    ap.add_argument("--arch", choices=["effidec3d", "swinv2", "smit"], default="effidec3d",
                    help="effidec3d = VoxelFox & VoCo (SwinV2 encoder, SSL-init only differs); smit = BSPC baseline")
    ap.add_argument("--roi", type=int, default=64, help="cubic crop side (voxels @1mm); must be divisible by 32")
    ap.add_argument("--a_min", type=float, default=0.0); ap.add_argument("--a_max", type=float, default=800.0)
    ap.add_argument("--levels", default="",
                    help="encoder output indices to concat ('set of 4'); default per arch: effidec x1..x4, smit 0..3")
    ap.add_argument("--pool", choices=["gap", "masked"], default="gap")
    ap.add_argument("--min_island_cc", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    cli = ap.parse_args()

    import pandas as pd
    assert cli.roi % 32 == 0, "EffiDec3D SwinV2 needs the crop side divisible by 32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, encoder_features = _build_model(cli)
    _assert_encoder_loaded(model, cli.base_ckpt, cli.arch)      # loud fail if encoder doesn't match
    load_base_ckpt(model, cli.base_ckpt)
    model = model.to(device).eval()
    level_idx = [int(x) for x in cli.levels.split(",")] if cli.levels else DEFAULT_LEVELS[cli.arch]

    tf = build_transform(SimpleNamespace(a_min=cli.a_min, a_max=cli.a_max))
    items = json.load(open(osp.join(cli.data_dir, cli.json_list)))[cli.mode]
    if cli.limit:
        items = items[:cli.limit]
    RD = osp.join("results", cli.results_dir, cli.modelname, cli.mode)
    roi = (cli.roi, cli.roi, cli.roi)

    rows, skipped = [], 0
    with torch.no_grad():
        for i, it in enumerate(items):
            key = _key_name_part(it["image"])
            if cli.contour == "gt":
                mask_path = gt_path(cli.data_dir, it)
                tmp_mask = None
            else:
                seg = osp.join(RD, f"{key}_seg.nii.gz")
                mask_path = _lcc_pred_path(seg, cli.min_island_cc) if osp.exists(seg) else None
                tmp_mask = mask_path
            if mask_path is None:
                skipped += 1; continue

            try:
                d = tf({"image": osp.join(cli.data_dir, it["image"]), "label": mask_path})
                img = d["image"].unsqueeze(0).float(); lab = d["label"].unsqueeze(0).float()   # (1,1,*V)
                fg = np.argwhere(lab[0, 0].cpu().numpy() > 0.5)
                if len(fg) == 0:
                    skipped += 1; continue
                ctr = fg.mean(0)
                ic, lc = roi_crop(img, lab, ctr, roi)                                           # (1,1,*roi)
                outs = encoder_features(ic.to(device))
                levels = [outs[j] for j in level_idx]
                vec = _pool_levels(levels, lc.to(device), cli.pool).cpu().numpy()

                row = {"id": key}
                row.update({f"feat_{j:04d}": float(v) for j, v in enumerate(vec)})
                rows.append(row)
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(items)} ...", flush=True)
            finally:
                _safe_unlink(tmp_mask)

    df = pd.DataFrame(rows)
    os.makedirs(osp.dirname(cli.out) or ".", exist_ok=True)
    df.to_csv(cli.out, index=False)
    print(f"\nsaved {len(df)} cases x {df.shape[1]-1} deep features ({cli.contour}, pool={cli.pool}) -> {cli.out}"
          f"  (skipped {skipped})")


if __name__ == "__main__":
    main()
