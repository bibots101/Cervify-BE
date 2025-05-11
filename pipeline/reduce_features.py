import joblib
import numpy as np
import pandas as pd
from utils.extract_utils import load_data, normalize_features

def reduce_with_pca(features: np.ndarray, pca_model) -> np.ndarray:
    return pca_model.transform(features)

def reduce_features(pca_model_path, scaler_path, features_df: pd.DataFrame) -> pd.DataFrame:
    pca_model = joblib.load(pca_model_path)
    scaler = joblib.load(scaler_path)

    features, handcrafted = load_data(features_df)
    normalized = normalize_features(features, scaler)
    reduced = reduce_with_pca(normalized, pca_model)

    reduced_df = pd.DataFrame(reduced, columns=[f'pca_{i}' for i in range(reduced.shape[1])])
    final_df = pd.concat([reduced_df, handcrafted.reset_index(drop=True)], axis=1)
    return final_df
