from medpy import metric
import numpy as np
from scipy import ndimage

import surface_distance
from surface_distance import metrics

LARGE = 9001

try:
    from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
    HAS_SURFACE_DISTANCE = True
except ImportError:
    HAS_SURFACE_DISTANCE = False
    print("Warning: surface_distance not installed. Surface DSC will be computed manually.")


def dice(input1, input2):
    return metric.dc(input1, input2)

def sum_dims(x, axis, dims):
    '''
    Given an array x, collapses dimensions listed in dims along the 
    specified axis, summing them together. Returns the reduced array.
    '''
    x = np.array(x)

    if len(dims) < 2:
        return x
    
    # Initialize output
    new_shape = list(x.shape)
    new_shape[axis] -= len(dims)-1
    x_ret = np.zeros(new_shape, dtype=x.dtype)
    
    # Sum over dims on axis
    sum_slices = [slice(None)]*x.ndim
    sum_slices[axis] = dims
    sum_slices = np.array(sum_slices, dtype=object)
    dim_sum = np.sum(x[sum_slices[0]], axis=axis, keepdims=True)
    
    # Remove all but first dim in dims
    mask = np.ones(x.shape, dtype=bool)
    mask_slices = [slice(None)]*x.ndim
    mask_slices[axis] = dims[1:]
    mask_slices = np.array(mask_slices, dtype=object)

    mask[mask_slices[0]] = 0

    x_ret.ravel()[...] = x[mask]
    
    # Put dim_sum into array at first dim
    replace_slices = [slice(None)]*x.ndim
    replace_slices[axis] = [dims[0]]
    replace_slices = np.array(replace_slices, dtype=object)
    x_ret[replace_slices[0]] = dim_sum
        
    return x_ret



def detect_lesions(prediction_mask, reference_mask, min_overlap=0.5):
    """
    Produces a mask for predicted lesions and a mask for reference lesions,
    with label IDs matching lesions together.
    
    Given a prediction and a reference mask, output a modified version of
    each where objects that overlap between the two mask share a label. This
    requires merging labels in the reference mask that are spanned by a single
    prediction and merging labels in the prediction mask that are spanned by
    a single reference. In cases where a label can be merged, separately, with
    more than one other label, a single merge option (label) is chosen to 
    accord the greatest overlap between the reference and prediction objects.
    
    After merging and matching, objects in the reference are considered
    detected if their respective predictions overlap them by more than
    `min_overlap` (intersection over union).
    
    :param prediction_mask: numpy.array
    :param reference_mask: numpy.array
    :param min_overlap: float in range [0, 1.]
    :return: prediction mask (int),
             reference mask (int),
             num_detected
    """
    
    # Initialize
    detected_mask = np.zeros(prediction_mask.shape, dtype=np.uint8)
    mod_reference_mask = np.copy(reference_mask)
    num_detected = 0
    if not np.any(reference_mask):
        return detected_mask, num_detected, 0, 0, 0
    
    if not min_overlap>0 and not min_overlap<=1:
        raise ValueError("min_overlap must be in [0, 1.]")
    
    # Get available IDs (excluding 0)
    # 
    # To reduce computation time, check only those lesions in the prediction 
    # that have any overlap with the ground truth.
    p_id_list = np.unique(prediction_mask[reference_mask.nonzero()])
    if p_id_list[0]==0:
        p_id_list = p_id_list[1:]
    g_id_list = np.unique(reference_mask)
    if g_id_list[0]==0:
        g_id_list = g_id_list[1:]
    
    # To reduce computation time, get views into reduced size masks.
    reduced_prediction_mask = rpm = prediction_mask.copy()
    for p_id in np.unique(prediction_mask):
        if p_id not in p_id_list and p_id!=0:
            reduced_prediction_mask[(rpm==p_id).nonzero()] = 0
    target_mask = np.logical_or(reference_mask, reduced_prediction_mask)
    bounding_box = ndimage.find_objects(target_mask)[0]
    r = reference_mask[bounding_box]
    p = prediction_mask[bounding_box]
    d = detected_mask[bounding_box]
    m = mod_reference_mask[bounding_box]

    # Compute intersection of predicted lesions with reference lesions.
    intersection_matrix = np.zeros((len(p_id_list), len(g_id_list)),
                                    dtype=np.int32)
    for i, p_id in enumerate(p_id_list):
        for j, g_id in enumerate(g_id_list):
            intersection = np.count_nonzero(np.logical_and(p==p_id, r==g_id))
            intersection_matrix[i, j] = intersection
            
    # Merge and label reference lesions that are connected by predicted
    # lesions.
    g_merge_count = dict([(g_id, 1) for g_id in g_id_list])
    for i, p_id in enumerate(p_id_list):
        # Identify g_id intersected by p_id
        g_id_indices = intersection_matrix[i].nonzero()[0]
        g_id_intersected = g_id_list[g_id_indices]
        
        # Make sure g_id are matched to p_id deterministically regardless of 
        # label order. Only merge those g_id which overlap this p_id more than
        # others.
        g_id_merge = []
        g_id_merge_indices = []
        for k, g_id in enumerate(g_id_intersected):
            idx = g_id_indices[k]
            if np.argmax(intersection_matrix[:, idx], axis=0)==i:
                # This g_id has the largest overlap with this p_id: merge.
                g_id_merge.append(g_id)
                g_id_merge_indices.append(idx)
                
        # Update merge count
        for g_id in g_id_merge:
            g_merge_count[g_id] = len(g_id_merge)
                
        # Merge. Update g_id_list, intersection matrix, mod_reference_mask.
        # Merge columns in intersection_matrix.
        g_id_list = np.delete(g_id_list, obj=g_id_merge_indices[1:])
        for g_id in g_id_merge:
            m[m==g_id] = g_id_merge[0]
        intersection_matrix = sum_dims(intersection_matrix,
                                        axis=1,
                                        dims=g_id_merge_indices)
    
    # Match each predicted lesion to a single (merged) reference lesion.
    max_val = np.max(intersection_matrix, axis=1)
    max_indices = np.argmax(intersection_matrix, axis=1)
    intersection_matrix[...] = 0
    intersection_matrix[np.arange(len(p_id_list)), max_indices] = max_val
    
    # Merge and label predicted lesions that are connected by reference
    # lesions.
    #
    # Merge rows in intersection_matrix.
    #
    # Here, it's fine to merge all p_id that are connected by a g_id since
    # each p_id has already been associated with only one g_id.
    for j, g_id in enumerate(g_id_list):
        p_id_indices = intersection_matrix[:,j].nonzero()[0]
        p_id_intersected = p_id_list[p_id_indices]
        intersection_matrix = sum_dims(intersection_matrix,
                                       axis=0,
                                       dims=p_id_indices)
        p_id_list = np.delete(p_id_list, obj=p_id_indices[1:])
        for p_id in p_id_intersected:
            d[p==p_id] = g_id
            
    # Trim away lesions deemed undetected.
    num_detected = len(p_id_list)
    for i, p_id in enumerate(p_id_list):
        for j, g_id in enumerate(g_id_list):
            intersection = intersection_matrix[i, j]
            if intersection==0:
                continue
            union = np.count_nonzero(np.logical_or(d==p_id, m==g_id))
            overlap_fraction = float(intersection)/union
            if overlap_fraction <= min_overlap:
                d[d==g_id] = 0      # Assuming one-to-one p_id <--> g_id
                num_detected -= g_merge_count[g_id]
                
    return detected_mask, mod_reference_mask, num_detected

def detect_lesions_2(prediction_mask, reference_mask, min_overlap=0.5):
    """
    Produces a mask for predicted lesions and a mask for reference lesions,
    with label IDs matching lesions together.
    
    Given a prediction and a reference mask, output a modified version of
    each where objects that overlap between the two mask share a label. This
    requires merging labels in the reference mask that are spanned by a single
    prediction and merging labels in the prediction mask that are spanned by
    a single reference. In cases where a label can be merged, separately, with
    more than one other label, a single merge option (label) is chosen to 
    accord the greatest overlap between the reference and prediction objects.
    
    After merging and matching, objects in the reference are considered
    detected if their respective predictions overlap them by more than
    `min_overlap` (intersection over union).
    
    :param prediction_mask: numpy.array
    :param reference_mask: numpy.array
    :param min_overlap: float in range [0, 1.]
    :return: prediction mask (int),
             reference mask (int),
             num_detected
    """
    
    # Initialize
    detected_mask = np.zeros(prediction_mask.shape, dtype=np.uint8)
    mod_reference_mask = np.copy(reference_mask)
    num_detected = 0
    if not np.any(reference_mask):
        return detected_mask, num_detected, 0, 0, 0
    
    if not min_overlap>0 and not min_overlap<=1:
        raise ValueError("min_overlap must be in [0, 1.]")
    
    # Get available IDs (excluding 0)
    # 
    # To reduce computation time, check only those lesions in the prediction 
    # that have any overlap with the ground truth.
    p_id_list = np.unique(prediction_mask[reference_mask.nonzero()])
    if p_id_list[0]==0:
        p_id_list = p_id_list[1:]
    g_id_list = np.unique(reference_mask)
    if g_id_list[0]==0:
        g_id_list = g_id_list[1:]
    
    # To reduce computation time, get views into reduced size masks.
    reduced_prediction_mask = rpm = prediction_mask.copy()
    for p_id in np.unique(prediction_mask):
        if p_id not in p_id_list and p_id!=0:
            reduced_prediction_mask[(rpm==p_id).nonzero()] = 0
    target_mask = np.logical_or(reference_mask, reduced_prediction_mask)
    bounding_box = ndimage.find_objects(target_mask)[0]
    r = reference_mask[bounding_box]
    p = prediction_mask[bounding_box]
    d = detected_mask[bounding_box]
    m = mod_reference_mask[bounding_box]

    # Compute intersection of predicted lesions with reference lesions.
    intersection_matrix = np.zeros((len(p_id_list), len(g_id_list)),
                                    dtype=np.int32)
    for i, p_id in enumerate(p_id_list):
        for j, g_id in enumerate(g_id_list):
            intersection = np.count_nonzero(np.logical_and(p==p_id, r==g_id))
            intersection_matrix[i, j] = intersection
            
    # Merge and label reference lesions that are connected by predicted
    # lesions.
    g_merge_count = dict([(g_id, 1) for g_id in g_id_list])
    # for i, p_id in enumerate(p_id_list):
    #     # Identify g_id intersected by p_id
    #     g_id_indices = intersection_matrix[i].nonzero()[0]
    #     g_id_intersected = g_id_list[g_id_indices]
        
    #     # Make sure g_id are matched to p_id deterministically regardless of 
    #     # label order. Only merge those g_id which overlap this p_id more than
    #     # others.
    #     g_id_merge = []
    #     g_id_merge_indices = []
    #     for k, g_id in enumerate(g_id_intersected):
    #         idx = g_id_indices[k]
    #         if np.argmax(intersection_matrix[:, idx], axis=0)==i:
    #             # This g_id has the largest overlap with this p_id: merge.
    #             g_id_merge.append(g_id)
    #             g_id_merge_indices.append(idx)
                
    #     # Update merge count
    #     for g_id in g_id_merge:
    #         g_merge_count[g_id] = len(g_id_merge)
                
    #     # Merge. Update g_id_list, intersection matrix, mod_reference_mask.
    #     # Merge columns in intersection_matrix.
    #     g_id_list = np.delete(g_id_list, obj=g_id_merge_indices[1:])
    #     for g_id in g_id_merge:
    #         m[m==g_id] = g_id_merge[0]
    #     intersection_matrix = sum_dims(intersection_matrix,
    #                                    axis=1,
    #                                    dims=g_id_merge_indices)
    
    # Match each predicted lesion to a single (merged) reference lesion.
    max_val = np.max(intersection_matrix, axis=1)
    max_indices = np.argmax(intersection_matrix, axis=1)
    intersection_matrix[...] = 0
    intersection_matrix[np.arange(len(p_id_list)), max_indices] = max_val
    
    # Merge and label predicted lesions that are connected by reference
    # lesions.
    #
    # Merge rows in intersection_matrix.
    #
    # Here, it's fine to merge all p_id that are connected by a g_id since
    # each p_id has already been associated with only one g_id.
    for j, g_id in enumerate(g_id_list):
        p_id_indices = intersection_matrix[:,j].nonzero()[0]
        p_id_intersected = p_id_list[p_id_indices]
        intersection_matrix = sum_dims(intersection_matrix,
                                       axis=0,
                                       dims=p_id_indices)
        p_id_list = np.delete(p_id_list, obj=p_id_indices[1:])
        for p_id in p_id_intersected:
            d[p==p_id] = g_id
            
    # Trim away lesions deemed undetected.
    num_detected = len(p_id_list)
    for i, p_id in enumerate(p_id_list):
        for j, g_id in enumerate(g_id_list):
            intersection = intersection_matrix[i, j]
            if intersection==0:
                continue
            union = np.count_nonzero(np.logical_or(d==p_id, m==g_id))
            overlap_fraction = float(intersection)/union
            if overlap_fraction <= min_overlap:
                d[d==g_id] = 0      # Assuming one-to-one p_id <--> g_id
                num_detected -= g_merge_count[g_id]
                
    return detected_mask, mod_reference_mask, num_detected


def compute_surface_dsc(pred_mask: np.ndarray, ref_mask: np.ndarray, 
                        voxel_spacing: tuple, tolerance_mm: float = 2.0) -> float:
    """
    Compute Normalized Surface Dice (Surface DSC) at a given tolerance.
    
    Args:
        pred_mask: Binary prediction mask
        ref_mask: Binary reference/ground truth mask
        voxel_spacing: Tuple of voxel dimensions (x, y, z) in mm
        tolerance_mm: Tolerance in mm for surface distance (default: 2.0mm)
    
    Returns:
        Surface DSC value between 0 and 1
    """
    pred_mask = pred_mask.astype(bool)
    ref_mask = ref_mask.astype(bool)
    
    # Handle edge cases
    if not pred_mask.any() and not ref_mask.any():
        return 1.0
    if not pred_mask.any() or not ref_mask.any():
        return 0.0
    
    if HAS_SURFACE_DISTANCE:
        # Use the surface_distance library
        surface_distances = compute_surface_distances(
            ref_mask, pred_mask, spacing_mm=voxel_spacing
        )
        surface_dsc = compute_surface_dice_at_tolerance(
            surface_distances, tolerance_mm=tolerance_mm
        )
    else:
        # Manual implementation using scipy
        from scipy.ndimage import distance_transform_edt, binary_erosion
        
        # Get surface voxels (boundary of the masks)
        def get_surface(mask):
            eroded = binary_erosion(mask)
            return mask & ~eroded
        
        pred_surface = get_surface(pred_mask)
        ref_surface = get_surface(ref_mask)
        
        if not pred_surface.any() or not ref_surface.any():
            return 0.0
        
        # Compute distance transforms
        dist_from_pred = distance_transform_edt(~pred_mask, sampling=voxel_spacing)
        dist_from_ref = distance_transform_edt(~ref_mask, sampling=voxel_spacing)
        
        # Get distances at surface points
        ref_surface_distances = dist_from_pred[ref_surface]
        pred_surface_distances = dist_from_ref[pred_surface]
        
        # Count surface points within tolerance
        ref_within_tolerance = np.sum(ref_surface_distances <= tolerance_mm)
        pred_within_tolerance = np.sum(pred_surface_distances <= tolerance_mm)
        
        total_surface_points = np.sum(ref_surface) + np.sum(pred_surface)
        
        surface_dsc = (ref_within_tolerance + pred_within_tolerance) / total_surface_points
    
    return float(surface_dsc)


def compute_volume_ratio(pred_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    """
    Compute volume ratio (predicted / reference).
    
    Values > 1 indicate over-segmentation, < 1 indicate under-segmentation.
    
    Args:
        pred_mask: Binary prediction mask
        ref_mask: Binary reference/ground truth mask
    
    Returns:
        Volume ratio (pred_volume / ref_volume)
    """
    pred_vol = pred_mask.astype(bool).sum()
    ref_vol = ref_mask.astype(bool).sum()
    
    if ref_vol == 0:
        return np.nan if pred_vol == 0 else np.inf
    
    return float(pred_vol / ref_vol)


def compute_segmentation_scores(prediction_mask, reference_mask,
                                voxel_spacing):
    """
    Calculates metrics scores from numpy arrays and returns an dict.
    
    Assumes that each object in the input mask has an integer label that 
    defines object correspondence between prediction_mask and 
    reference_mask.
    
    :param prediction_mask: numpy.array, int
    :param reference_mask: numpy.array, int
    :param voxel_spacing: list with x,y and z spacing
    :return: dict with dice, jaccard, voe, rvd, assd, rmsd, and msd
    """
    
    scores = {'dice': [],
              'hd95': [],
              'sdsc': [],
              'volratio': []
              }
    
    for i, obj_id in enumerate(np.unique(prediction_mask)):
        if obj_id==0:
            continue    # 0 is background, not an object; skip

        # Limit processing to the bounding box containing both the prediction
        # and reference objects.
        target_mask = (reference_mask==obj_id)+(prediction_mask==obj_id)
        bounding_box = ndimage.find_objects(target_mask)[0]
        p = (prediction_mask==obj_id)[bounding_box]
        r = (reference_mask==obj_id)[bounding_box]
        if np.any(p) and np.any(r):
            dice = metric.dc(p,r)
            # jaccard = dice/(2.-dice)
            scores['dice'].append(dice)
            
            surface_distances = surface_distance.compute_surface_distances(p, r, spacing_mm=voxel_spacing)

            distmetric = metrics.compute_robust_hausdorff(surface_distances, 95)
            scores['hd95'].append(distmetric)
            
            scores['sdsc'].append(compute_surface_dsc(p, r, voxel_spacing=voxel_spacing, tolerance_mm=2))
            scores['volratio'].append(compute_volume_ratio(p, r))
            
        else:
            # There are no objects in the prediction, in the reference, or both
            scores['dice'].append(np.nan)
            scores['hd95'].append(np.nan)
            scores['sdsc'].append(np.nan)
            scores['volratio'].append(np.nan)
    return scores