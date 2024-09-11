import os
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define paths
input_dirs = {
    'train': './data_set/train',
    'validation': './data_set/validation',
    'test': './data_set/test'
}

output_dir = './data_set/working'

for subset in input_dirs.keys():
    subset_dir = os.path.join(output_dir, subset)
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)

def resize_and_save_image(input_path, output_path, size=(128, 128)):
  try:
    with Image.open(input_path) as img:
      if img.mode == 'P':
          img = img.convert('RGBA')

      if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
          img = img.convert('RGB')

      img = img.resize(size, Image.LANCZOS)
      img.save(output_path, format='JPEG')
  except Exception as e:
    print(f"Error processing {input_path}: {e}")

def process_directory(input_directory, output_directory):
  for root, dirs, files in os.walk(input_directory):
    relative_path = os.path.relpath(root, input_directory)
    output_path = os.path.join(output_directory, relative_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in files:
      if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
          input_file_path = os.path.join(root, file_name)
          output_file_path = os.path.join(output_path, file_name)
          resize_and_save_image(input_file_path, output_file_path)

# Process each directory separately
for subset, dir_path in input_dirs.items():
    process_directory(dir_path, os.path.join(output_dir, subset))

print("Resizing and saving images completed.")
