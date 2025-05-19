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
if datasetType == 'jd':

	with open('static/JD_data/img2product_illegal.json', 'r') as json_file:
					img2products = json.load(json_file)
	file_path ='static/JD_data/products_illegal.json'
	with open(file_path, 'r') as json_file:
					products = json.load(json_file)				