import os
import joblib
import numpy as np
import pandas as pd
from keras.models import load_model
from utils.extract_utils import normalize_features
from utils.global_var import SCALER_MLP, PCA_MODEL_10CLASSES, MLP_CLASSIFIER, SCALER_PCA_10CLASSES
import absl.logging

absl.logging.set_verbosity('error')

handcrafted_features = [
    'nc_ratio', 'cell_area', 'circularity', 'aspect_ratio',
    'mean_gray', 'std_gray', 'edge_density', 'mean_R', 'mean_G', 'mean_B'
]

mlp_model = load_model(MLP_CLASSIFIER)
scaler_classifier = joblib.load(SCALER_MLP)
scaler_pca = joblib.load(SCALER_PCA_10CLASSES)
pca_model = joblib.load(PCA_MODEL_10CLASSES)

id_to_label = {
    0: "Candidose",
    1: "Trichomonas",
    2: "ASC-US",
    3: "ASC-H",
    4: "LSIL",
    5: "SCC",
    6: "AGC",
    7: "Actinomyces",
    8: "Flora",
    9: "Herpes"
}

def classify_mlp(image_name, svm_df, features_df):
    mask = svm_df["svm_label"] == "Other"
    if not mask.any():
        return pd.DataFrame()
    
    mlp_features = features_df[mask]
    dino_only = mlp_features.drop(columns=handcrafted_features)
    handcrafted = mlp_features[handcrafted_features]

    normalized = normalize_features(dino_only, scaler_pca)
    reduced = pca_model.transform(normalized)

    combined = pd.DataFrame(reduced, columns=[f'pca_{i}' for i in range(reduced.shape[1])])
    combined = pd.concat([combined, handcrafted.reset_index(drop=True)], axis=1)

    X = normalize_features(combined, scaler_classifier)
    preds = mlp_model.predict(X, verbose=0)

    class_ids = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    labels = [id_to_label[i] for i in class_ids]

    result_df = svm_df[mask].copy()
    result_df["mlp_class"] = class_ids
    result_df["mlp_label"] = labels
    result_df["mlp_confidence"] = confidences

    return result_df
