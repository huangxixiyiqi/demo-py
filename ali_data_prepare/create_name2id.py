import json
import os
from tqdm import tqdm
def is_exist(files):
    for i in range(0,9):
        file_path = os.path.join('/home/hl/code/demo/static/ali_data',f'train_text_img_pairs_{i}_compressed', files)
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                    return os.path.join(f'train_text_img_pairs_{i}_compressed', files)

def create_name2id_dict():
    train_cap_file = '/home/hl/code/demo/static/ali_data/train.json'
    val_cap_file = '/home/hl/code/demo/static/ali_data/val.json'
    image2id={}
    id2image={}
    image_counter = 0
    with open(train_cap_file, 'r', encoding='utf-8')as fp:
        datas = json.load(fp)
        for i in tqdm(datas):
            image = i['product']
            imagePath = is_exist(image)
            image2id[imagePath] = image_counter
            id2image[image_counter] = imagePath
            image_counter += 1
    
    with open(val_cap_file, 'r', encoding='utf-8')as fp:
        datas = json.load(fp)
        for i in tqdm(datas):
            image = i['product']
            imagePath = is_exist(image)
            image2id[imagePath] = image_counter
            id2image[image_counter] = imagePath
            image_counter += 1

    with open('image2id.json', 'w') as json_file:
        json.dump(image2id, json_file, indent=4)
    with open('id2image.json', 'w') as json_file:
            json.dump(id2image, json_file, indent=4)
    print(image_counter)
create_name2id_dict()