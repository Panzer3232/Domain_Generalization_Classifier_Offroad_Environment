import os
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from skimage import draw, io
from matplotlib import pyplot as plt
from scipy import stats
from collections import defaultdict
import tensorflow as tf

# Set up TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Base path and domain
base_path = '/home/ashri/transformations/crops'
domain = 'fall_crops'
src_folder = os.path.join(base_path, domain, 'tree')

# Output folders
dest_folder_photo = os.path.join(base_path, domain, 'photo')
dest_folder_sketch = os.path.join(base_path, domain, 'sketch')
dest_folder_art_painting = os.path.join(base_path, domain, 'art_painting')
dest_folder_cartoons = os.path.join(base_path, domain, 'cartoons')

# Create output directories if they don't exist
os.makedirs(dest_folder_photo, exist_ok=True)
os.makedirs(dest_folder_sketch, exist_ok=True)
os.makedirs(dest_folder_art_painting, exist_ok=True)
os.makedirs(dest_folder_cartoons, exist_ok=True)

# List all image files in the source folder
image_files = os.listdir(src_folder)
total_images = len(image_files)


# Calculate the number of images for each class
num_cartoons = int(0.3 * total_images)
num_sketch = int(0.25 * total_images)
num_art_painting = int(0.15 * total_images)
num_photo = int(0.3 * total_images)

# Shuffle images and split them into the respective classes
random.shuffle(image_files)
cartoons_files = image_files[:num_cartoons]
sketch_files = image_files[num_cartoons:num_cartoons + num_sketch]
art_painting_files = image_files[num_cartoons + num_sketch:num_cartoons + num_sketch + num_art_painting]
photo_files = image_files[num_cartoons + num_sketch + num_art_painting:]

# Function to copy images to the photo class folder
def copy_images_to_folder(file_list, dest_folder):
    for image_file in file_list:
        src_file = os.path.join(src_folder, image_file)
        dest_file = os.path.join(dest_folder, image_file)
        shutil.copy2(src_file, dest_file)

# Oilify transformation function (art_painting)
def oilify_image(input_image, brush_size=5.0, expression_level=2.0):
    brushSizeInt = int(brush_size)
    expressionSize = brush_size * expression_level
    margin = int(expressionSize * 2)
    halfBrushSizeInt = brushSizeInt // 2
    
    shape = ((input_image.shape[0] - 2 * margin) // brushSizeInt, (input_image.shape[1] - 2 * margin) // brushSizeInt)
    brushes = [draw.ellipse(halfBrushSizeInt, halfBrushSizeInt, brush_size, random.randint(brushSizeInt, expressionSize), rotation=random.random() * np.pi) for _ in range(50)]

    result = np.zeros(input_image.shape, dtype=np.uint8)

    for x in range(margin, input_image.shape[0] - margin, brushSizeInt):
        for y in range(margin, input_image.shape[1] - margin, brushSizeInt):
            ellipseXs, ellipseYs = random.choice(brushes)
            result[x + ellipseXs, y + ellipseYs] = input_image[x, y]
    
    return result

# Cartoon transformation function
def caart(img):
    kernel = np.ones((2, 2), np.uint8)
    output = np.array(img)
    x, y, c = output.shape
    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], 5, 150, 150)
    
    edge = cv2.Canny(output, 100, 200)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
    
    hists = []
    
    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256 + 1))
    hists.append(hist)
    
    C = []
    for h in hists:
        C.append(K_histogram(h))
    
    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))
    output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
    
    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, 0, thickness=1)
    
    for i in range(3):
        output[:, :, i] = cv2.erode(output[:, :, i], kernel, iterations=1)
    
    return output

def K_histogram(hist):
    alpha = 0.001
    N = 80
    C = np.array([128])
    
    while True:
        C, groups = update_c(C, hist)
        
        new_C = set()
        for i, indice in groups.items():
            if len(indice) < N:
                new_C.add(C[i])
                continue
            
            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                left = 0 if i == 0 else C[i - 1]
                right = len(hist) - 1 if i == len(C) - 1 else C[i + 1]
                delta = right - left
                if delta >= 3:
                    c1 = (C[i] + left) / 2
                    c2 = (C[i] + right) / 2
                    new_C.add(c1)
                    new_C.add(c2)
                else:
                    new_C.add(C[i])
            else:
                new_C.add(C[i])
        
        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))
    
    return C

def update_c(C, hist):
    while True:
        groups = defaultdict(list)
        
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(C - i)
            index = np.argmin(d)
            groups[index].append(i)
        
        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice * hist[indice]) / np.sum(hist[indice]))
        
        if np.sum(new_C - C) == 0:
            break
        C = new_C
    
    return C, groups

# Sketch transformation function using TensorFlow for GPU acceleration
def sketch_transform_tf(img):
    gray_image = tf.image.rgb_to_grayscale(img)
    inverted_gray_image = 255 - gray_image
    blurred_image = tf.image.adjust_brightness(inverted_gray_image, 21.0)
    inverted_blurred_image = 255 - blurred_image
    pencil_sketch_image = tf.divide(gray_image, inverted_blurred_image) * 256.0
    return pencil_sketch_image

# Function to save transformed images
def save_transformed_image(input_image_path, output_image_path, transform_function, use_gpu=False):
    input_image = cv2.imread(input_image_path)
    if use_gpu:
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        result = transform_function(input_image)
        result = result.numpy().astype(np.uint8)
    else:
        result = transform_function(input_image)
    cv2.imwrite(output_image_path, result)

# Copy images to the photo class folder
copy_images_to_folder(photo_files, dest_folder_photo)

# Apply sketch transformation and save to sketch class folder
for image_file in sketch_files:
    src_file = os.path.join(src_folder, image_file)
    dest_file = os.path.join(dest_folder_sketch, image_file)
    save_transformed_image(src_file, dest_file, sketch_transform_tf, use_gpu=True)

# Apply oilify transformation and save to art_painting class folder
for image_file in art_painting_files:
    src_file = os.path.join(src_folder, image_file)
    dest_file = os.path.join(dest_folder_art_painting, image_file)
    save_transformed_image(src_file, dest_file, oilify_image)

# Apply cartoon transformation and save to cartoon class folder
for image_file in cartoons_files:
    src_file = os.path.join(src_folder, image_file)
    dest_file = os.path.join(dest_folder_cartoons, image_file)
    save_transformed_image(src_file, dest_file, caart)

print("Dataset processing complete.")
