import pandas as pd

handcrafted_features = ['nc_ratio', 'cell_area', 'circularity', 'aspect_ratio',
                        'mean_gray', 'std_gray', 'edge_density', 'mean_R', 'mean_G', 'mean_B']

def load_data(df):
    features_pca = df.drop(columns=handcrafted_features)
    handcrafted = df[handcrafted_features]
    return features_pca.values, handcrafted

def save_to_csv(reduced_features, handcrafted, output_path):
    reduced_df = pd.DataFrame(reduced_features, columns=[f'pca_{i}' for i in range(reduced_features.shape[1])])
    final_df = pd.concat([reduced_df, handcrafted.reset_index(drop=True)], axis=1)
    final_df.to_csv(output_path, index=False)

def normalize_features(features,scaler):
    return scaler.transform(features)