import torch
import clip
from PIL import Image
from tqdm import tqdm
print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
import os
import h5py
images_path = '/media/disk2/zz/msr_image_data'
feature_file = h5py.File('./msr_all_clip-B16_feat.hdf5','w')
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



