import faiss
import numpy as np
import h5py

with h5py.File('/home/hl/code/demo/msr_all_clip_feat_concat.h5', 'r') as f:
			all_image_feat = f['all_clip_feat_concat'][:]
   
   