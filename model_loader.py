# model_loader.py
from ultralytics import YOLO

def load_yolo_model(model_path="yolo11s.pt"):
    """Incarca modelul YOLO si efectueaza fuziunea pentru a optimiza performanta de inferenta."""
    model = YOLO(model_path)
    # model.fuse()  # Fuzioneaza straturile pentru imbunatatirea performantei
    return model
