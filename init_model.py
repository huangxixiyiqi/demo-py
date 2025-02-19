'''
Author: huangxixi huangxixiyiqi@gmail.com
Date: 2024-11-24 09:26:24
LastEditors: huangxixi huangxixiyiqi@gmail.com
LastEditTime: 2025-02-19 10:45:21
FilePath: /demo/init_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from CLIP.clip import clip
import torch
import cn_clip.clip as cn_clip
from cn_clip.clip import load_from_name
from whoosh.index import create_in, open_dir
import json
datasetType = "jd" # ali msr jd


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
if datasetType == 'jd':
	state_dict = torch.load("CLIP/ViT-B_32_finetuned.pt", map_location="cpu")
	clip_model.load_state_dict(state_dict)
	clip_model.to(device)
 


cnclip_model, _ = load_from_name("ViT-B-16", device=device, download_root='./')
if datasetType == 'jd':
	state_dict = torch.load("Chinese_CLIP/CNCLIP_ViT-B_16_finetuned.pt", map_location="cpu")
	cnclip_model.load_state_dict(state_dict, strict=False)
	cnclip_model.to(device)
cnclip_model.eval()



with open('static/JD_data/img2product.json', 'r') as json_file:
				img2products = json.load(json_file)
with open('static/JD_data/products.json', 'r') as json_file:
				products = json.load(json_file)				