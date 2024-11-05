import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import clip
from PIL import Image
from tqdm import tqdm
print("Available models:", available_models())  
# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
import os
import h5py
images_path = '/media/disk2/zz/msr_image_data'
feature_file = h5py.File('./msr_all_cnclip_feat.hdf5','w')
for file in tqdm(os.listdir(images_path)):
    image_path = os.path.join(images_path,file)
    # print(image_path)
    # break
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image).cpu().numpy()
        feature_file.create_dataset(file,data=image_feature)
feature_file.close()
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)



