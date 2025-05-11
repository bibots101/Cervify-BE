import joblib
import numpy as np
import pandas as pd
from utils.global_var import SVM_CLASSIFIER, SCALER_SVM
from utils.extract_utils import normalize_features

svm_model = joblib.load(SVM_CLASSIFIER)
scaler = joblib.load(SCALER_SVM)

id_to_label = {
    0: "NILM",
    1: "HSIL",
    2: "Other"
}

def classify_svm(image_name, features_df, segmentation_df):
    X = normalize_features(features_df, scaler)
    probs = svm_model.predict_proba(X)
    preds = np.argmax(probs, axis=1)
    result_df = segmentation_df.copy()
    result_df["svm_class"] = preds
    result_df["svm_label"] = [id_to_label[i] for i in preds]
    result_df["svm_confidence"] = [probs[i, cls] for i, cls in enumerate(preds)]
    return result_df
