import h5py
import numpy as np
image_clip_feat = h5py.File('/home/zz/code/zz_code/demo/msr_all_clip-B16_feat.hdf5')
# image_clip_feat = h5py.File('/home/zz/code/zz_code/demo/msr_part_clip_feat.hdf5')

image_keys = list(image_clip_feat.keys())
all_image_feat = []
for key in image_keys:
    all_image_feat.append(image_clip_feat[key][:])
all_image_feat = np.concatenate(all_image_feat, axis=0)

print(all_image_feat.shape)
with h5py.File('msr_all_clip-B16_feat_concat.h5', 'w') as f:
    # 创建数据集
    f.create_dataset('all_clip_feat_concat', data=all_image_feat)