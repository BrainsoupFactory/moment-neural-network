# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:42:27 2024

@author: qiyangku
"""
import h5py
import numpy as np

def calculate_degree():
    single_voxel_size = 2500
    n_region = 378
    degree = 500
    brain_file = h5py.File(r".\projects\dtb\data\DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat", "r")
    # brain_file = h5py.File(
    #     r'/public/home/ssct004t/project/wangjiexiang/criticality_test/JFFengdata/DTB_3_res_2mm_Region_Data_Perpared_Feb07_24.mat')
    sc = np.array(brain_file['Region_Streamline'])
    Region_Label = np.array(brain_file['Region_Label']).reshape(-1)
    
    sc = sc[:n_region, :n_region]
    uni_label = np.unique(Region_Label)
    uni_label = np.delete(uni_label, (0, 379, 380))[:n_region]
    
    Voxel_GM = np.array(brain_file['Voxel_GM']).reshape(-1)
    Region_GM = np.zeros(uni_label.shape[0])
    for _relative_id_ in range(uni_label.shape[0]):
        _select_voxel_idxes = np.where(Region_Label == uni_label[_relative_id_])[0]
        Region_GM[_relative_id_] = Voxel_GM[_select_voxel_idxes].sum()
    regionwize_NSR_dti_grey_matter = Region_GM
    block_size = regionwize_NSR_dti_grey_matter / regionwize_NSR_dti_grey_matter.sum() * single_voxel_size * n_region
    block_size = np.maximum(block_size, single_voxel_size / 10)  # TODO
    # block_size = np.maximum(block_size, 250) # TODO
    popu_size = (block_size[:, None] * np.array([0.8, 0.2])[None, :]).reshape(
        [-1]).astype(np.int64)
    
    # compute edges (not degree)
    # inner E: inner I: outer E = 0.6 : 0.15 : 0.25# degree ratio: fixed degree for each neuron within the same voxel
    total_edges = block_size.sum() * degree  # n_region * single_voxel_size * degree
    edegs_in_per_voxel = sc.sum(axis=1) / np.sum(sc) * total_edges
    
    degree_ = np.maximum(edegs_in_per_voxel / block_size, degree / 7)
    degree_ = np.minimum(degree_, degree * 7)  # TODO
    
    # below are custom edits by Qi 
    # local intra-regional connections
    K_EE = degree_*0.6
    K_EI = degree_*0.15
    K_IE = degree_*0.6
    K_II = degree_*0.15
    
    # long range inter-regional excitatory connections
    K_EE_long = degree_*0.25
    K_EE_long = K_EE_long*sc/sc.sum(axis=1, keepdims=True)
    np.fill_diagonal(K_EE_long, 0.0) # make sure self connection excluded
    
    return K_EE, K_EI, K_IE, K_II, K_EE_long

if __name__=='__main__':    
    K_EE, K_EI, K_IE, K_II, K_EE_long = calculate_degree()


