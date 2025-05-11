import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolo", "yolo_model.pt")
DINO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "dinov2_vitl14_full.pt")

SCALER_PCA_3CLASSES = os.path.join(ROOT_DIR, "models", "SVM", "scaler_pca.joblib")
PCA_MODEL_3CLASSES = os.path.join(ROOT_DIR, "models", "SVM", "pca_model.joblib")
SCALER_SVM = os.path.join(ROOT_DIR, "models", "SVM", "svm_scaler.joblib")
SVM_CLASSIFIER = os.path.join(ROOT_DIR, "models", "SVM", "svm_model.joblib")

MLP_CLASSIFIER = os.path.join(ROOT_DIR, "models", "mlp", "mlp_model.keras")
PCA_MODEL_10CLASSES = os.path.join(ROOT_DIR, "models", "mlp", "pca_model.joblib")
SCALER_PCA_10CLASSES = os.path.join(ROOT_DIR, "models", "mlp", "scaler_pca.joblib")
SCALER_MLP = os.path.join(ROOT_DIR, "models", "mlp", "scaler_classifer.joblib")

CRYPT_KEY = "6Ggd16zLYwTabLpIEDWfrliEsXuu85owWJ6uwF8hLos='"