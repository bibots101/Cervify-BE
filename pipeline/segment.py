from ultralytics import YOLO
from utils.image_loader import load_image
from utils.preprocess import mobile_preprocess
import pandas as pd
from utils.global_var import YOLO_MODEL_PATH

model = None
def load_yolo():
    global model
    if model is None:
        model = YOLO(YOLO_MODEL_PATH)
    return model

def segment_image(image_path,type):
    image = load_image(image_path)
    if type == "mobile":
        image = mobile_preprocess(image)
    w, h = image.size
    model = load_yolo()
    results = model.predict(image_path, imgsz=640, conf=0.25, verbose=False)
    r = results[0] if isinstance(results, list) else results
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    class_names = [r.names[cid] for cid in class_ids]
    if boxes.size == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame({
        "class_id": class_ids,
        "class_name": class_names,
        "confidence": scores,
        "x1": boxes[:, 0],
        "y1": boxes[:, 1],
        "x2": boxes[:, 2],
        "y2": boxes[:, 3],
        "image_path": r.path
    })

    if df[["x1", "y1", "x2", "y2"]].isnull().values.any() or df.shape[0] == 0:
        return pd.DataFrame()
    
    return df