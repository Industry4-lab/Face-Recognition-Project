
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2, 
    height_shift_range=0.2,  
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    fill_mode='nearest'
)

input_dir = "C:/Users/ksiri/OneDrive/Desktop/images" # Path to the main directory containing subfolders 
output_dir = "C:/Users/ksiri/OneDrive/Desktop/agumented images 1"  # Path to save augmented images 

os.makedirs(output_dir, exist_ok=True)

for subdir, dirs, files in os.walk(input_dir): 
    for dir_name in dirs:
        input_subdir = os.path.join(subdir, dir_name) 
        output_subdir = os.path.join(output_dir, dir_name) 
        os.makedirs(output_subdir, exist_ok=True)
        for file_name in os.listdir(input_subdir):
            file_path = os.path.join(input_subdir, file_name)
            if file_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')): 
                img = load_img(file_path)
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                aug_iter = datagen.flow(img_array, batch_size=1, save_to_dir=output_subdir, save_prefix='aug', 
                                        save_format='jpeg')
                for i in range(10):  # Change 10 to the desired number of augmented images
                    next(aug_iter)
