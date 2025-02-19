import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from cn_clip.clip import load_from_name, available_models
import cn_clip.clip as clip
from PIL import Image
from tqdm import tqdm
import h5py
import json
import numpy as np

import torch
# 打印可用模型
print("Available models:", available_models())

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
state_dict = torch.load("/home/hl/code/demo/Chinese_CLIP/CNCLIP_ViT-B_16_finetuned.pt", map_location="cpu")
load_info = model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()
id2text = {}
# 读取 txt 文件
with open('./static/JD_data/jd_text_name.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        cid = line.split(' ', 1)[0]
        cap = line.split(' ', 1)[1]
        id2text[cid] = cap


feature_file = h5py.File('./image_features/jd_all_text_name_cnclip-B16_feat_concate.hdf5', 'w')

# 设置批量大小
batch_size = 1024
text_batch = []
text_ids = list(id2text.keys())
num_text = len(text_ids)

# 预分配特征数组
features = []

for i in tqdm(range(0, num_text, batch_size), desc="抽取cnclip特征"):
    batch_ids = text_ids[i:i+batch_size]
    text_batch_ = [id2text[text_id] for text_id in batch_ids]
    text_batch = clip.tokenize(text_batch_).to(device)
    
    # # 将单张图片的张量合并成一个批次
    # image_batch = torch.cat(image_batch).to(device)
    
    # 处理批次
    with torch.no_grad():
        batch_features = model.encode_text(text_batch).cpu().numpy()
        features.append(batch_features)

# 将所有特征拼接在一起
features = np.concatenate(features, axis=0)

# 保存特征到 HDF5 文件
feature_file.create_dataset('all_clip_feat_concat', data=features)
feature_file.close()

