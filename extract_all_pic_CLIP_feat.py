import torch
import clip
from PIL import Image
from tqdm import tqdm
import os
import h5py
import json
import numpy as np

print(clip.available_models())

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
state_dict = torch.load("/home/zms/code/old/ali_race_codes/ali_race_clip/experiment/4000000_ali_race_data/CLIP freeze_layer_num_0/runs_0_400W/ViT-B_32_finetuned.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.to(device)
# 读取 JSON 文件
with open('/media/disk2/hl/code/demo/static/JD_data/Index2Image.json', 'r') as json_file:
    id2image = json.load(json_file)

images_path = '/media/disk2/hl/code/demo/static/JD_data/Images'
feature_file = h5py.File('./image_features/jd_all_clip-B32_feat_concate.hdf5', 'w')

# 设置批量大小
batch_size = 1024
image_batch = []
image_ids = list(id2image.keys())
num_images = len(image_ids)

# 预分配特征数组
features = []

for i in tqdm(range(0, num_images, batch_size), desc="抽取clip特征"):
    batch_ids = image_ids[i:i+batch_size]
    image_batch = [preprocess(Image.open(os.path.join(images_path, id2image[img_id]))).unsqueeze(0) for img_id in batch_ids]
    
    # 将单张图片的张量合并成一个批次
    image_batch = torch.cat(image_batch).to(device)
    
    # 处理批次
    with torch.no_grad():
        batch_features = model.encode_image(image_batch).cpu().numpy()
        features.append(batch_features)

# 将所有特征拼接在一起
features = np.concatenate(features, axis=0)

# 保存特征到 HDF5 文件
feature_file.create_dataset('all_clip_feat_concat', data=features)
feature_file.close()
