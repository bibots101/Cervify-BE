import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
import pandas as pd
from utils.global_var import DINO_MODEL_PATH
import logging

logging.getLogger('torch').setLevel(logging.ERROR)


_model = None
device = None

transform = transforms.Compose([
    transforms.Resize((518, 518), interpolation=TF.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_dino_model():
    global _model
    global device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _model is None:
        model = torch.load(DINO_MODEL_PATH, map_location=device)
        model.eval().to(device)
        _model = model
    return _model

def extract_handcrafted_features(crop):
    gray_np = np.array(crop.convert("L"))
    rgb_np = np.array(crop.convert("RGB"))
    threshold = np.percentile(gray_np, 20)
    nucleus_area = np.sum(gray_np < threshold)
    cytoplasm_area = gray_np.size - nucleus_area
    nc_ratio = nucleus_area / cytoplasm_area if cytoplasm_area else 0
    cell_area = gray_np.size
    circularity = 1.0
    aspect_ratio = crop.size[0] / (crop.size[1] + 1e-5)
    mean_gray = np.mean(gray_np)
    std_gray = np.std(gray_np)
    edges = cv2.Canny(gray_np, 100, 200)
    edge_density = np.sum(edges) / cell_area
    r_mean = np.mean(rgb_np[:, :, 0])
    g_mean = np.mean(rgb_np[:, :, 1])
    b_mean = np.mean(rgb_np[:, :, 2])
    return [round(nc_ratio, 4), round(cell_area, 2), round(circularity, 4), round(aspect_ratio, 4),
            round(mean_gray, 2), round(std_gray, 2), round(edge_density, 4),
            round(r_mean, 2), round(g_mean, 2), round(b_mean, 2)]

def extract_dino_features(crop):
    global device
    global _model
    _model = get_dino_model()
    crop_tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        features = _model(crop_tensor).cpu().numpy().flatten()
    return features

def extract_features(image_path, segment_df):
    features_list = []
    image = Image.open(image_path).convert("RGB")
    for _, row in segment_df.iterrows():
        x1, y1, x2, y2 = map(float, [row["x1"], row["y1"], row["x2"], row["y2"]])
        if x2 - x1 <= 5 or y2 - y1 <= 5:
            continue
        crop = image.crop((x1, y1, x2, y2))
        dino_feats = extract_dino_features(crop)
        handcrafted_feats = extract_handcrafted_features(crop)
        combined_feats = dino_feats.tolist() + handcrafted_feats
        features_list.append(combined_feats)
    header = [f'feature_{i}' for i in range(1024)] + [
        'nc_ratio', 'cell_area', 'circularity', 'aspect_ratio',
        'mean_gray', 'std_gray', 'edge_density', 'mean_R', 'mean_G', 'mean_B'
    ]
    return pd.DataFrame(features_list, columns=header)
