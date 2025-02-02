import numpy as np
import cv2
import os
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from tqdm import tqdm

def compute_glcm_descriptor(image_path):
    # Read and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Normalize image to reduce number of gray levels (to 32 levels)
    image = ((image / 255.0) * 31).astype(np.uint8)
    
    # Configuration for GLCM
    distances = [1, 2, 4, 8]  # Similar to scales in GIST
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 4 angles
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    
    features = []
    
    # Divide image into 4x4 grid (like GIST)
    block_h = image.shape[0] // 4
    block_w = image.shape[1] // 4
    
    for i in range(4):
        for j in range(4):
            # Extract block
            block = image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            
            # Compute GLCM for each distance
            for distance in distances:
                glcm = graycomatrix(block, 
                                  distances=[distance], 
                                  angles=angles,
                                  levels=32, 
                                  symmetric=True, 
                                  normed=True)
                
                # Compute GLCM properties
                for prop in properties:
                    feature = graycoprops(glcm, prop).flatten()
                    features.extend(feature)
    
    return np.array(features)

def process_directory(base_path):
    all_features = []
    all_labels = []
    
    # First, collect all valid image paths
    image_paths = []
    for root, dirs, files in os.walk(base_path):
        if 'Negative' in root:
            continue
        category = os.path.basename(root)
        for file in files:
            if file.endswith(('.png')):
                image_paths.append((os.path.join(root, file), category))
    
    for image_path, category in tqdm(image_paths, desc="Processing GLCM"):
        features = compute_glcm_descriptor(image_path)
        if features is not None:
            all_features.append(features)
            all_labels.append(category)
    
    df = pd.DataFrame(all_features)
    df['category'] = all_labels
    df['label'] = df['category'].astype('category').cat.codes
    return df

if __name__ == "__main__":
    base_path = '/home/duyle/Downloads/Rice_photos'
    df = process_directory(base_path)
    df.to_csv('rice_glcm_features.csv', index=False)
