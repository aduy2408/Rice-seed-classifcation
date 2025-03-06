#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import os
import pandas as pd
from skimage.measure import moments, moments_central, shannon_entropy
from sklearn.cluster import KMeans
import mahotas
from tqdm import tqdm
from scipy.stats import skew, kurtosis
import pywt


# In[1]:


def compute_zernike_moments(image, degree=8): # 25 features
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    _, new_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(new_img.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], contourIdx=-1, color=(255,), thickness=-1)
    binary = (mask > 0).astype(np.uint8)
    
    (x, y), radius = cv2.minEnclosingCircle(contour)
    x, y = int(x), int(y)
    radius = int(np.ceil(radius))
    x1 = max(x - radius, 0)
    y1 = max(y - radius, 0)
    x2 = x + radius
    y2 = y + radius
    
    cropped_mask = binary[y1:y2, x1:x2]
    
    h, w = cropped_mask.shape
    if h != w:
        size = max(h, w)
        square_mask = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square_mask[y_offset:y_offset+h, x_offset:x_offset+w] = cropped_mask
    else:
        square_mask = cropped_mask
        
    effective_radius = square_mask.shape[0] // 2
    zernike_moments = mahotas.features.zernike_moments(square_mask, effective_radius, degree)
    
    return {f"Zernike_{i}": zernike_moments[i] for i in range(len(zernike_moments))}


# In[2]:


def compute_central_moments(image): # 16 features
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float64)
    M = moments(image)
    mm_central = moments_central(image, center=(M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])).flatten()
    return {f"Central_{i}": mm_central[i] for i in range(len(mm_central))}


# In[3]:


def compute_lbp_feature(image): # 10 features
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(image, R=1, P=8, method="uniform")
    hist, bins = np.histogram(lbp.flatten(), bins=10, range=(0, 10))
    return {f"LBP_{i}": hist[i] for i in range(len(hist))}


# In[4]:


def compute_texture_feature(image): # 4 features
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256), density=True)
    bins = bins[:-1]
    mean = np.sum(hist * bins)
    std = np.sqrt(np.sum((bins - mean)**2 * hist))
    uniformity = np.sum(hist**2)
    third_moment = np.sum((bins - mean)**3 * hist)
    return {
        "texture_mean": mean,
        "texture_std": std,
        "texture_uniformity": uniformity,
        "texture_third_moment": third_moment
    }


# In[5]:


def compute_color_feature(image): # 33 features
    def entropy(channel):
        return shannon_entropy(channel)
    def waveLet(channel):
        max_level = pywt.dwt_max_level(min(channel.shape), "db4")
        coeffs = pywt.wavedec2(channel, "db4", level=max_level)
        return np.mean(coeffs[0].ravel())
    
    # BGR
    B, G, R = cv2.split(image)
    mean_R, mean_G, mean_B = np.mean(R), np.mean(G), np.mean(B)
    # sqrt_R, sqrt_G, sqrt_B = np.sqrt(mean_R), np.sqrt(mean_G), np.sqrt(mean_B)
    std_R, std_G, std_B = np.std(R), np.std(G), np.std(B)
    skew_R, skew_G, skew_B = skew(R.flatten()), skew(G.flatten()), skew(B.flatten())
    kur_R, kur_G, kur_B = kurtosis(R.flatten()), kurtosis(G.flatten()), kurtosis(B.flatten())
    ent_R, ent_G, ent_B = entropy(R), entropy(G), entropy(B)
    wav_R, wav_G, wav_B = waveLet(R), waveLet(G), waveLet(B)
    
    # HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    mean_h, mean_s, mean_v = np.mean(h), np.mean(s), np.mean(v)
    std_h, std_s, std_v = np.std(h), np.std(s), np.std(v)
    # sqrt_h, sqrt_s, sqrt_v = np.sqrt(mean_h), np.sqrt(mean_s), np.sqrt(mean_v)
    skew_h, skew_s, skew_v = skew(h.flatten()), skew(s.flatten()), skew(v.flatten())
    kur_h, kur_s, kur_v = kurtosis(h.flatten()), kurtosis(s.flatten()), kurtosis(v.flatten())
    ent_h, ent_s, ent_v = entropy(h), entropy(s), entropy(v)
    wav_h, wav_s, wav_v = waveLet(h), waveLet(s), waveLet(v)
    
    # Lab
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    mean_l, mean_a, mean_b = np.mean(l), np.mean(a), np.mean(b)
    std_l, std_a, std_b = np.std(l), np.std(a), np.std(b)
    # sqrt_l, sqrt_a, sqrt_b = np.sqrt(mean_l), np.sqrt(mean_a), np.sqrt(mean_b)
    skew_l, skew_a, skew_b = skew(l.flatten()), skew(a.flatten()), skew(b.flatten())
    kur_l, kur_a, kur_b = kurtosis(l.flatten()), kurtosis(a.flatten()), kurtosis(b.flatten())
    ent_l, ent_a, ent_b = entropy(l), entropy(a), entropy(b)
    wav_l, wav_a, wav_b = waveLet(l), waveLet(a), waveLet(b)
    
    # YCbCr
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)
    mean_y, mean_cr, mean_cb = np.mean(y), np.mean(cr), np.mean(cb)
    std_y, std_cr, std_cb = np.std(y), np.std(cr), np.std(cb)
    # sqrt_y, sqrt_cr, sqrt_cb = np.sqrt(mean_y), np.sqrt(mean_cr), np.sqrt(mean_cb)
    skew_y, skew_cr, skew_cb = skew(y.flatten()), skew(cr.flatten()), skew(cb.flatten())
    kur_y, kur_cr, kur_cb = kurtosis(y.flatten()), kurtosis(cr.flatten()), kurtosis(cb.flatten())
    ent_y, ent_cr, ent_cb = entropy(y), entropy(cr), entropy(cb)
    wav_y, wav_cr, wav_cb = waveLet(y), waveLet(cr), waveLet(cb)
    
    # XYZ
    xyz_img = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    X, Y, Z = cv2.split(xyz_img)
    mean_X, mean_Y, mean_Z = np.mean(X), np.mean(Y), np.mean(Z)
    std_X, std_Y, std_Z = np.std(X), np.std(Y), np.std(Z)
    skew_X, skew_Y, skew_Z = skew(X.flatten()), skew(Y.flatten()), skew(Z.flatten())
    kur_X, kur_Y, kur_Z = kurtosis(X.flatten()), kurtosis(Y.flatten()), kurtosis(Z.flatten())
    ent_X, ent_Y, ent_Z = entropy(X), entropy(Y), entropy(Z)
    wav_X, wav_Y, wav_Z = waveLet(X), waveLet(Y), waveLet(Z)
    
    return {"mean_r": mean_R, "mean_g": mean_G, "mean_B": mean_B,
            # "sqrt_r": sqrt_R, "sqrt_g": sqrt_G, "sqrt_B": sqrt_B,
            "std_r": std_R, "std_g": std_G, "std_B": std_B,
            "skew_r": skew_R, "skew_g": skew_G, "skew_B": skew_B,
            "kur_r": kur_R, "kur_g": kur_G, "kur_B": kur_B,
            "ent_r": ent_R, "ent_g": ent_G, "ent_B": ent_B,
            "wav_r": wav_R, "wav_g": wav_G, "wav_B": wav_B,
            
            "mean_h": mean_h, "mean_s": mean_s, "mean_v": mean_v,
            "std_h": std_h, "std_s": std_s, "std_v": std_v,
            # "sqrt_h": sqrt_h, "sqrt_s": sqrt_s, "sqrt_v": sqrt_v,
            "skew_h": skew_h, "skew_s": skew_s, "skew_v": skew_v,
            "kur_h": kur_h, "kur_s": kur_s, "kur_v": kur_v,
            "ent_h": ent_h, "ent_s": ent_s, "ent_v": ent_v,
            "wav_h": wav_h, "wav_s": wav_s, "wav_v": wav_v,
            
            "mean_l": mean_l, "mean_a": mean_a, "mean_b": mean_b,
            "std_l": std_l, "std_a": std_a, "std_b": std_b,
            # "sqrt_l": sqrt_l, "sqrt_a": sqrt_a, "sqrt_b": sqrt_b,
            "skew_l": skew_l, "skew_a": skew_a, "skew_b": skew_b,
            "kur_l": kur_l, "kur_a": kur_a, "kur_b": kur_b,
            "ent_l": ent_l, "ent_a": ent_a, "ent_b": ent_b,
            "wav_l": wav_l, "wav_a": wav_a, "wav_b": wav_b,
            
            "mean_y": mean_y, "mean_cb": mean_cb, "mean_cr": mean_cr,
            "std_y": std_y, "std_cb": std_cb, "std_cr": std_cr,
            # "sqrt_y": sqrt_y, "sqrt_cb": sqrt_cb, "sqrt_cr": sqrt_cr,
            "skew_y": skew_y, "skew_cb": skew_cb, "skew_cr": skew_cr,
            "kur_y": kur_y, "kur_cb": kur_cb, "kur_cr": kur_cr,
            "ent_y": ent_y, "ent_cb": ent_cb, "ent_cr": ent_cr,
            "wav_y": wav_y, "wav_cb": wav_cb, "wav_cr": wav_cr,
            
            "mean_x": mean_X, "mean_Y": mean_Y, "mean_z": mean_Z,
            "std_x": std_X, "std_Y": std_Y, "std_z": std_Z,
            "skew_x": skew_X, "skew_Y": skew_Y, "skew_z": skew_Z,
            "kur_x": kur_X, "kur_Y": kur_Y, "kur_z": kur_Z,
            "ent_x": ent_X, "ent_Y": ent_Y, "ent_z": ent_Z,
            "wav_x": wav_X, "wav_Y": wav_Y, "wav_z": wav_Z
            }


# In[6]:


def basic_feature(image): # 8 features
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    threshold, new_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    area = np.count_nonzero(new_img)
    contours, _ = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    
    length = x + w
    width = y + h
    ratio = length / width
    
    ellipse = cv2.fitEllipse(contour)
    
    major_axis = max(ellipse[1])
    minor_axis = min(ellipse[1])
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)
    
    sf1 = major_axis / area
    sf2 = minor_axis / area
    sf3 = area / ((0.5 * major_axis)**2 * np.pi)
    sf4 = area / (0.5**2 * major_axis * minor_axis * np.pi)
    
    ed = np.sqrt(4 * area / np.pi)
    ar = major_axis / minor_axis
    roundness = (4 * area * np.pi) / peri**2
    Co = ed / major_axis
    solid = area / hull_area
    
    return {
        "area": area,
        "length": length,
        "width": width,
        "ratio": ratio,
        "major_axis_length": major_axis,
        "minor_axis_length": minor_axis,
        "convex_hull_area": hull_area,
        "convex_hull_perimeter": hull_perimeter,
        "shape_factor_1": sf1,
        "shape_factor_2": sf2,
        "shape_factor_3": sf3,
        "shape_factor_4": sf4,
        "equivalent_diameter": ed,
        "aspect_ratio": ar,
        "perimeter": peri,
        "roundness": roundness,
        "compactness": Co,
        "solidity": solid
    }


# In[7]:


def compute_glcm_descriptor(image): # 16 features
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Configuration 
    distance = [3]  
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
    properties = ['contrast', 'correlation', 'energy', 'homogeneity']  
    
    glcm = graycomatrix(image, distances=distance, angles=angles, symmetric=True, normed=True)
    
    features = []
    for prop in properties:
        feature = graycoprops(glcm, prop).flatten()
        features.extend(feature)
    
    return np.array(features)


# In[8]:


def compute_percentile_features(image):
    features = {}
    
    percentiles = [5, 25, 50, 75, 95]
    
    for color_space, prefix, channel_names in [
        (None, 'bgr', ['B', 'G', 'R']),  
        (cv2.COLOR_BGR2HSV, 'hsv', ['H', 'S', 'V']),
        (cv2.COLOR_BGR2LAB, 'lab', ['L', 'A', 'B']),
        (cv2.COLOR_BGR2YCrCb, 'ycrcb', ['Y', 'Cr', 'Cb']),
        (cv2.COLOR_BGR2XYZ, 'xyz', ['X', 'Y', 'Z'])
    ]:
        if color_space is None:
            converted = image  
        else:
            converted = cv2.cvtColor(image, color_space)
        
        channels = cv2.split(converted)
        
        for i, channel_name in enumerate(channel_names):
            for p in percentiles:
                value = np.percentile(channels[i], p)
                features[f'pf_{prefix}_{channel_name.lower()}_p{p}'] = value
    
    return features

def compute_color_variance_ratios(image):
    features = {}
    
    # BGR
    B, G, R = cv2.split(image)
    var_B, var_G, var_R = np.var(B), np.var(G), np.var(R)
    
    # Variance ratios
    features['var_ratio_R_G'] = var_R / (var_G + 1e-7)
    features['var_ratio_R_B'] = var_R / (var_B + 1e-7)
    features['var_ratio_G_B'] = var_G / (var_B + 1e-7)
    
    # HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    var_H, var_S, var_V = np.var(H), np.var(S), np.var(V)
    
    features['var_ratio_H_S'] = var_H / (var_S + 1e-7)
    features['var_ratio_H_V'] = var_H / (var_V + 1e-7)
    features['var_ratio_S_V'] = var_S / (var_V + 1e-7)
    
    # Lab
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L, A, B_lab = cv2.split(lab_img)
    var_L, var_A, var_B_lab = np.var(L), np.var(A), np.var(B_lab)
    
    features['var_ratio_L_A'] = var_L / (var_A + 1e-7)
    features['var_ratio_L_B'] = var_L / (var_B_lab + 1e-7)
    features['var_ratio_A_B'] = var_A / (var_B_lab + 1e-7)
    
    return features

def compute_color_range_features(image):
    features = {}
    
    for color_space, prefix, channel_names in [
        (None, 'bgr', ['B', 'G', 'R']),  
        (cv2.COLOR_BGR2HSV, 'hsv', ['H', 'S', 'V']),
        (cv2.COLOR_BGR2LAB, 'lab', ['L', 'A', 'B']),
        (cv2.COLOR_BGR2YCrCb, 'ycrcb', ['Y', 'Cr', 'Cb']),
        (cv2.COLOR_BGR2XYZ, 'xyz', ['X', 'Y', 'Z'])
    ]:
        if color_space is None:
            converted = image  
        else:
            converted = cv2.cvtColor(image, color_space)
        
        channels = cv2.split(converted)
        
        for i, channel_name in enumerate(channel_names):
            channel = channels[i]
            
            # Range 
            features[f'range_{prefix}_{channel_name.lower()}_range'] = np.max(channel) - np.min(channel)
            
            # Interquartile range 
            features[f'iqr_{prefix}_{channel_name.lower()}_iqr'] = np.percentile(channel, 75) - np.percentile(channel, 25)

            
            # Mode + mode concentration
            hist, bin_edges = np.histogram(channel, bins=256, range=(0, 256))
            mode_bin = np.argmax(hist)
            mode_value = (bin_edges[mode_bin] + bin_edges[mode_bin + 1]) / 2
            mode_concentration = hist[mode_bin] / np.sum(hist)
            
            features[f'mv_{prefix}_{channel_name.lower()}_mode'] = mode_value
            features[f'mc_{prefix}_{channel_name.lower()}_mode_conc'] = mode_concentration
    
    return features


# In[9]:


def mask_image(image):
    original = image.copy()  

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)  

    threshold, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.mean(gray[binary == 255]) > np.mean(gray[binary == 0]):
        binary = cv2.bitwise_not(binary)
        
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, (255), -1)  

    masked_image = cv2.bitwise_and(original, original, mask=mask)
    return masked_image


# In[10]:


def extract_all_features(image):
    # mask = mask_image(image)
    features = {}

    lbp_hist = compute_lbp_feature(image)
    features.update(lbp_hist) 
    
    texture_features = compute_texture_feature(image)
    features.update(texture_features) 

    color_features = compute_color_feature(image)
    features.update(color_features) 

    # central_features = compute_central_moments(image)
    # features.update(central_features)
    
    zernike_features = compute_zernike_moments(image, degree=8)
    features.update(zernike_features)
    
    shape_features = basic_feature(image)
    features.update(shape_features)

    glcm_features = compute_glcm_descriptor(image)
    if glcm_features is not None:
        for i, val in enumerate(glcm_features):
            features[f'GLCM_{i}'] = val
        

    
    var_color = compute_color_variance_ratios(image)
    features.update(var_color)
    
    percentile_features = compute_percentile_features(image)
    features.update(percentile_features)
    
    color_range_features = compute_color_range_features(image)
    features.update(color_range_features)
    

    return features


# In[12]:


def process_directory(base_path):
    all_data = []
    image_paths = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                label = 0 if 'negative' in root else 1
                image_paths.append((os.path.join(root, file), label))
    
    for image_path, label in tqdm(image_paths, desc="Processing Images"):
        image = cv2.imread(image_path)
        if image is None:
            continue  

        features = extract_all_features(image)
        features["Label"] = label
 
        all_data.append(features)
    
    data = pd.DataFrame(all_data)
    
    return data


# In[ ]:


def process_directory(base_path):
    all_data = []
    image_paths = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                label = 0 if 'Negative' in root else 1
                image_paths.append((os.path.join(root, file), label))

    for image_path, label in tqdm(image_paths, desc="Processing Images"):
        image = cv2.imread(image_path)
        if image is None:
            continue  

        features = extract_all_features(image)
        features["Label"] = label
 
        all_data.append(features)

    df = pd.DataFrame(all_data)

    return df


# In[ ]:


# df = process_directory("/home/duyle/Documents/AIL/rice_seed/Xi-23")


# In[ ]:


# df.to_csv('test_xi-23.csv',index=False)


# In[14]:


types = ['BC-15','Huong_thom-1','Nep-87','Q-5','Thien_uu-8','Xi-23', 'TBR-36', 'TBR_45', 'TH3-5']
for type in types:
    df = process_directory(f'/home/duyle/Documents/AIL/rice_seed/{type}')
    df.to_csv(f'all_with_zernike_pluscolor_enhanced_edge1_{type}.csv',index=False)


# In[ ]:


# types = ['BC-15','Huongthom','Nep87','Q5','Thien_uu','Xi23']
# for type in types:
#     df = process_directory(f'/home/duyle/Documents/AIL/Rice_photos-master/{type}')
#     df.to_csv(f'/home/duyle/Rice_photos/results/test_old_{type}.csv',index=False)

