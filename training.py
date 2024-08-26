import cv2
import os

root_folder = "./data_set/Training"

def load_images_from_folder(root_folder):
    images = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # Add more formats if needed
                img_path = os.path.join(subdir, file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
    return images

images = load_images_from_folder(root_folder)
print(images[0])
