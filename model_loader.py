# model_loader.py

from ultralytics import YOLO

MODEL_PATH = "yolo26s.pt"   # modelul folosit de aplicatie (se descarca automat la prima rulare)


def load_yolo_model(model_path=MODEL_PATH):
    """Incarca modelul YOLO pre-antrenat pe setul de date COCO."""
    model = YOLO(model_path)
    return model
