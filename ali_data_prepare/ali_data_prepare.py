import os
from PIL import Image
from tqdm import tqdm

def rename_and_save_images(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_counter = 0
    
    
    for folder_name in tqdm(os.listdir(root_dir), desc="Processing Images"):
        folder_path = os.path.join(root_dir, folder_name)

        if os.path.isdir(folder_path) and ('compressed' in folder_path or 'val_imgs' in folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # 检查是否为图片文件（扩展名判断）
                if image_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    try:
                        # 打开并保存图片到新的文件夹，并重命名为从0开始的序号
                        img = Image.open(image_path)
                        img = img.convert("RGB")  # 确保统一格式
                        output_path = os.path.join(output_dir, f"{image_counter}.jpg")
                        img.save(output_path, "JPEG")
                        image_counter += 1
                    except Exception as e:
                        print(f"无法处理图片 {image_path}：{e}")
    print(image_counter)
                    

# 使用示例
root_dir = "/home/hl/code/demo/static/ali_data"  # 替换为包含图片的文件夹路径
output_dir = "/media/disk2/hl/VisualSearch/ali_data/ali_images"  # 替换为重命名后图片的保存路径
rename_and_save_images(root_dir, output_dir)
