import os
import pandas as pd

def merge_final_results(image_name, svm_df, mlp_df):
    os.makedirs("data/debug", exist_ok=True)
    final_path = f"data/debug/{image_name}.csv"

    final_df = svm_df[[
        "image_path", "x1", "y1", "x2", "y2", "svm_class", "svm_label", "svm_confidence"
    ]].copy()

    if not mlp_df.empty:
        keys = ["x1", "y1", "x2", "y2"]
        final_df = final_df.merge(
            mlp_df[keys + ["mlp_class", "mlp_label", "mlp_confidence"]],
            on=keys,
            how="left"
        )
    else:
        final_df["mlp_class"] = None
        final_df["mlp_label"] = None
        final_df["mlp_confidence"] = None

    final_df.to_csv(final_path, index=False)
    return final_df
