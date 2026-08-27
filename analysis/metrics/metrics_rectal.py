import os
import os.path as osp
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for lits_metric_utils
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import nibabel as nib

from scipy.ndimage import label as label_connected_components
from scipy.ndimage import generate_binary_structure

import json

import pandas as pd
import argparse

from lits_metric_utils import detect_lesions, detect_lesions_2, compute_segmentation_scores


def get_gt_path(data_dir, item, mode):
    """
    Ground-truth path. Prefer an explicit 'label' in the manifest (now present for the testing split too);
    fall back to the imagesTs->labelsTs / strip-_z800 convention for image-only manifests (older jsons,
    other cohorts). Both resolve to the SAME file, so this does not change existing results.
    `mode` retained for signature compatibility.
    """
    if item.get('label'):
        return osp.join(data_dir, item['label'])
    return osp.join(data_dir, item['image']).replace("imagesTs/", "labelsTs/").replace("_z800", "")


def get_pred_path(results_dir, modelname, gt_filename, mode, is_nnunet=False):
    """Get prediction path based on mode and model type."""
    if modelname == 'nnunet':
        return osp.join('results', modelname, mode, gt_filename)
    else:
        return osp.join('results', results_dir, modelname, mode,
                        gt_filename.replace(".nii.gz", "_z800_seg.nii.gz"))


def evaluate_case(gt_path, pred_path, surface_tolerance_mm=2.0):
    """Evaluate a single case and return scores."""
    file_gt_nii_img = nib.load(gt_path)
    voxel_spacing = file_gt_nii_img.header.get_zooms()[:3]
    voxel_vol_cc = np.prod(voxel_spacing) / 1000.0
    gt_volume = file_gt_nii_img.get_fdata().astype(np.int8)

    # Process ground truth - remove small components
    ref_mask_lesion, num_reference = label_connected_components(
        gt_volume == 1,
        structure=generate_binary_structure(3, 3),
        output=np.int16
    )

    component_counts = np.bincount(ref_mask_lesion.flatten())
    valid_components = np.where(component_counts > 10)[0]
    valid_components = valid_components[valid_components != 0]
    gt_volume2 = np.where(np.isin(ref_mask_lesion, valid_components), ref_mask_lesion, 0)

    ref_mask_lesion, num_reference = label_connected_components(
        gt_volume2 > 0,
        structure=generate_binary_structure(3, 3),
        output=np.int16
    )

    ref_counts = np.bincount(ref_mask_lesion.ravel())
    ref_volumes_cc = [c * voxel_vol_cc for i, c in enumerate(ref_counts) if i != 0 and c > 0]

    if num_reference == 0:
        return {
            'dice': np.nan,
            'hd95': np.nan,
            'surface_dsc': np.nan,
            'volume_ratio': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'pred_volumes_cc': np.nan,
            'ref_volumes_cc': np.nan
        }

    # Load and process prediction
    file_pred_nii_img = nib.load(pred_path)
    pred_volume = file_pred_nii_img.get_fdata().astype(np.int8)

    pred_mask_lesion, num_predicted = label_connected_components(
        pred_volume == 1,
        structure=generate_binary_structure(3, 3),
        output=np.int16
    )

    component_counts = np.bincount(pred_mask_lesion.flatten())
    valid_components = np.where(component_counts > 10)[0]
    pred_volume2 = np.where(np.isin(pred_mask_lesion, valid_components), pred_mask_lesion, 0)

    pred_mask_lesion, num_predicted = label_connected_components(
        pred_volume2,
        structure=generate_binary_structure(3, 3),
        output=np.int16
    )

    pred_counts = np.bincount(pred_mask_lesion.ravel())
    pred_volumes_cc = [c * voxel_vol_cc for i, c in enumerate(pred_counts) if i != 0 and c > 0]

    # Detect lesions
    try:
        detected_mask_lesion, mod_ref_mask, num_detected = detect_lesions(
            prediction_mask=pred_mask_lesion,
            reference_mask=ref_mask_lesion,
            min_overlap=0.1
        )
    except ValueError:
        detected_mask_lesion, mod_ref_mask, num_detected = detect_lesions_2(
            prediction_mask=pred_mask_lesion,
            reference_mask=ref_mask_lesion,
            min_overlap=0.1
        )

    # Compute scores
    score_TP = num_detected
    score_FP = num_predicted - num_detected
    score_FN = num_reference - num_detected

    score_precision = score_TP / (score_TP + score_FP + 1e-11)
    score_recall = score_TP / (score_TP + score_FN + 1e-11)

    lesion_scores = compute_segmentation_scores(
        prediction_mask=detected_mask_lesion,
        reference_mask=mod_ref_mask,
        voxel_spacing=voxel_spacing
    )
    
    return {
        'dice': np.nanmean(lesion_scores['dice']),
        'hd95': np.nanmean(lesion_scores['hd95']),
        'surface_dsc': np.nanmean(lesion_scores['sdsc']),
        'volume_ratio': np.nanmean(lesion_scores['volratio']),
        'precision': score_precision,
        'recall': score_recall,
        'pred_volumes_cc': np.mean(pred_volumes_cc) if pred_volumes_cc else 0.0,
        'ref_volumes_cc': np.mean(ref_volumes_cc) if ref_volumes_cc else 0.0,
    }


def process_single_case(args_tuple):
    """Wrapper for parallel processing."""
    gt_path, pred_path, gt_filename, surface_tolerance_mm = args_tuple
    try:
        scores = evaluate_case(gt_path, pred_path, surface_tolerance_mm)
        scores['name'] = gt_filename
        return scores
    except Exception as e:
        return {
            'name': gt_filename,
            'dice': np.nan,
            'hd95': np.nan,
            'surface_dsc': np.nan,
            'volume_ratio': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'pred_volumes_cc': np.nan,
            'ref_volumes_cc': np.nan,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")
    parser.add_argument("--modelname", default='swinunetr', help="Name of the model")
    parser.add_argument("--mode", default='testing', choices=['validation', 'testing'],
                        help="Evaluation mode: validation or testing")
    parser.add_argument("--data_dir", default='data_rectal', help="Ground truth data directory")
    parser.add_argument("--json_file", default='Trainval_set1.json', help="JSON file with dataset split")
    parser.add_argument("--results_dir", default='rectal_swinunetr_96x96x96_base', help="Results directory")
    parser.add_argument("--output_dir", default='analysis/csvs', help="Output directory for CSV")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of parallel workers (default: CPU count)")
    parser.add_argument("--surface_tolerance", default=2.0, type=float, 
                        help="Tolerance in mm for Surface DSC (default: 2.0mm)")

    args = parser.parse_args()
    # --json_file may be a bare filename living inside --data_dir, or a path that
    # already resolves on its own. Only join when it is the former.
    if not osp.isabs(args.json_file) and not osp.exists(args.json_file):
        args.json_file = osp.join(args.data_dir, args.json_file)


    if args.num_workers is None:
        args.num_workers = mp.cpu_count()
        
    print(args)
    
    # Load dataset
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    workingset = data[args.mode]

    print(f"Evaluating: {args.modelname}")
    print(f"Mode: {args.mode}")
    print(f"Cases: {len(workingset)}")
    print(f"Results dir: {args.results_dir}")
    print(f"Workers: {args.num_workers}")
    print(f"Surface DSC tolerance: {args.surface_tolerance}mm")
    print("-" * 50)

    # Prepare arguments for parallel processing
    task_args = []
    for item in workingset:
        gt_path = get_gt_path(args.data_dir, item, args.mode)
        gt_filename = osp.basename(gt_path)
        pred_path = get_pred_path(args.results_dir, args.modelname, gt_filename, args.mode)
        task_args.append((gt_path, pred_path, gt_filename, args.surface_tolerance))

    # Run in parallel
    score_records = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_case, arg): arg for arg in task_args}
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if 'error' in result:
                print(f"Error processing {result['name']}: {result['error']}")
                del result['error']
            score_records.append(result)

    # Save results
    df = pd.DataFrame(score_records)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.modelname == 'nnunet':
        output_file = osp.join(args.output_dir, f"{args.modelname.replace('/', '_')}_{args.mode}.csv")
    else:
        output_file = osp.join(args.output_dir, f"{args.results_dir.replace('/', '_')}_{args.mode}.csv")

    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
