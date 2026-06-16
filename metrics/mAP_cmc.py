import torch
import numpy as np

def mAP_cmc(dist_mat, query_aids, gallery_aids, query_cams, gallery_cams, max_rank = 50):
    """
    Compute mean AP and CMC scores.
    
    Args:
        dist_mat (np.ndarray): Distance matrix with shape (num_query, num_gallery).
        query_aids (np.ndarray): 1-D array containing animal IDs of each query.
        gallery_aids (np.ndarray): 1-D array containing animal IDs of each gallery.
        query_cams (np.ndarray): 1-D array containing camera IDs of each query.
        gallery_cams (np.ndarray): 1-D array containing camera IDs of each gallery.
        max_rank (int, optional): Maximum CMC rank to be computed. Default is 50.
    
    Returns:
        all_cmc (np.ndarray): CMC score at each rank up to the maximum CMC rank.
        mAP (np.ndarray): The mAP metric.
    """
    num_query, num_gallery = dist_mat.shape
    assert num_query == len(query_aids), f"Number of queries ({len(query_aids)}) should be the same as the distance matrix ({num_query})."
    assert num_gallery == len(gallery_aids), f"Number of galleries ({len(gallery_aids)}) should be the same as the distance matrix ({num_gallery})."
    if num_gallery < max_rank:
        max_rank = num_gallery
        print(f"The number of gallery samples ({num_gallery}) is smaller than the maximum CMC rank.")
    
    indices = np.argsort(dist_mat, axis = 1)    # Shape: (num_query, num_gallery)
    match = (gallery_aids[indices] == query_aids[:, np.newaxis]).astype(np.int32)    # Shape: (num_query, num_gallery)

    all_cmc = []
    all_AP = []
    num_valid_queries = 0
    for q_idx in range(num_query):
        q_aid = query_aids[q_idx]
        q_cam = query_cams[q_idx]
        q_orig_cmc = match[q_idx, :]    # binary indicator of correct matches, Shape: (num_gallery,)
        
        if not np.any(q_orig_cmc):
            print(f"There is no gallery sample with the ID: {q_aid}.")
            continue
        
        # Compute CMC curve.
        cmc = np.cumsum(q_orig_cmc)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])    # Shape: (max_rank,)
        num_valid_queries += 1
        
        # Compute mAP.
        num_rel = np.sum(q_orig_cmc)    # number of positive galleries for the given query
        tmp_cmc = np.cumsum(q_orig_cmc)
        count_gallery = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        precision = tmp_cmc / count_gallery
        AP = np.sum(q_orig_cmc * precision) / num_rel
        all_AP.append(AP)
    
    if num_valid_queries == 0:
        raise RuntimeError("All query identities do not appear in gallery.")
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)    # Shape: (num_query, max_rank)
    all_cmc = np.sum(all_cmc, axis = 0) / num_valid_queries    # Shape: (max_rank,)
    mAP = np.mean(all_AP)

    return all_cmc, mAP


