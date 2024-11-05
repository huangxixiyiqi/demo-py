import torch
from cn_clip.clip import load_from_name, available_models
from PIL import Image
from tqdm import tqdm
import os
import h5py
import json
import numpy as np

# 打印可用模型
print("Available models:", available_models())

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()

# 读取 JSON 文件
with open('/home/hl/code/demo/static/ali_data/id2image.json', 'r') as json_file:
    id2image = json.load(json_file)

images_path = '/home/hl/code/demo/static/ali_data'
feature_file = h5py.File('./image_features/ali_all_cnclip-B16_feat_concate.hdf5', 'w')

# 设置批量大小
batch_size = 1024
image_ids = list(id2image.keys())
num_images = len(image_ids)

# 创建 HDF5 数据集时不预先分配大小
feature_dataset = feature_file.create_dataset('all_clip_feat_concat', shape=(0, 512), maxshape=(None, 512), dtype=np.float32)

for i in tqdm(range(0, num_images, batch_size), desc="抽取cnclip特征"):
    batch_ids = image_ids[i:i + batch_size]
    image_batch = [preprocess(Image.open(os.path.join(images_path, id2image[img_id]))).unsqueeze(0) for img_id in batch_ids]
    
    # 合并批次
    image_batch = torch.cat(image_batch).to(device)
    
    with torch.no_grad():
        batch_features = model.encode_image(image_batch).cpu().numpy()
        
        # 扩展数据集大小并保存当前批次特征
        feature_dataset.resize(feature_dataset.shape[0] + batch_features.shape[0], axis=0)
        feature_dataset[-batch_features.shape[0]:] = batch_features

feature_file.close()
