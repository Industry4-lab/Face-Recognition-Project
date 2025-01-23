
import os
from PIL import Image

dataset_path = "C:/Users/ksiri/OneDrive/Desktop/agumented images for face recongition" 
output_path = "C:/Users/ksiri/OneDrive/Desktop/Resized images"

new_size = (224, 224)

if not os.path.exists(output_path):
    os.makedirs(output_path)

def resize_images_in_folder(input_folder, output_folder, new_size): 
    for root, _, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder) 
        output_subfolder = os.path.join(output_folder, relative_path)
        if not os.path.exists(output_subfolder): 
            os.makedirs(output_subfolder)
        for file in files:
            if file.endswith(".jpeg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img: 
                    img_resized = img.resize(new_size)
                    img_resized.save(os.path.join(output_subfolder, file))

resize_images_in_folder(dataset_path, output_path, new_size)
print("Resizing images in all subfolders complete.")
