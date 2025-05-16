import argparse
import os
import sys
import warnings
import pandas as pd
from pipeline.segment import segment_image
from pipeline.feature_extractor import extract_features
from pipeline.reduce_features import reduce_features
from pipeline.classify_stage1 import classify_svm
from pipeline.classify_stage2 import classify_mlp
from pipeline.merge_final_results import merge_final_results
from utils.progress_tracker import set_progress
from utils.global_var import SCALER_PCA_3CLASSES, PCA_MODEL_3CLASSES

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_pipeline(image_path,type):
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    set_progress(image_name, "Segmenting image...", 10)
    
    segment_df = segment_image(image_path,type)
    if segment_df.empty:
        set_progress(image_name, "Segmentation failed.", 10)
        return pd.DataFrame()

    set_progress(image_name, "Extracting features...", 20)
    features_df = extract_features(image_path, segment_df)

    set_progress(image_name, "Reducing features...", 40)
    final_df = reduce_features(PCA_MODEL_3CLASSES, SCALER_PCA_3CLASSES, features_df)
    if final_df.empty:
        set_progress(image_name, "Feature reduction failed.", 40)
        return pd.DataFrame()

    set_progress(image_name, "Classifying with SVM...", 60)
    svm_df = classify_svm(image_name, final_df, segment_df)

    if (svm_df["svm_label"] == "Other").any():
        set_progress(image_name, "Classifying with MLP...", 75)
        mlp_df = classify_mlp(image_name, svm_df, features_df)
    else:
        mlp_df = pd.DataFrame()
        set_progress(image_name, "MLP stage skipped.", 75)

    set_progress(image_name, "Merging results...", 85)
    resultat = merge_final_results(image_name, svm_df, mlp_df)
    if resultat.empty:
        set_progress(image_name, "Merging failed.", 85)
        return pd.DataFrame()

    set_progress(image_name, "Finalizing results...", 99)
    resultat["label"] = resultat.apply(
        lambda row: row["mlp_label"] if row["svm_label"] == "Other" else row["svm_label"],
        axis=1
    )
    resultat["confidence"] = resultat.apply(
        lambda row: row["mlp_confidence"] if row["svm_label"] == "Other" else row["svm_confidence"],
        axis=1
    )
    resultat["class_id"] = resultat.apply(
        lambda row: row["mlp_class"] if row["svm_label"] == "Other" else row["svm_class"],
        axis=1
    )

    final_columns = ["image_path", "x1", "y1", "x2", "y2", "class_id", "label", "confidence"]
    resultat = resultat[final_columns]

    os.makedirs("data/results", exist_ok=True)
    final_path = f"data/results/{image_name}.csv"
    resultat.to_csv(final_path, index=False)

    set_progress(image_name, "Done.", 100)
    return resultat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the Pap smear image")
    args = parser.parse_args()

    run_pipeline(args.image)