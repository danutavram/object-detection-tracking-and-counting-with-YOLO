# vehicle_detector.py
from model_loader import load_yolo_model

def detect_vehicles(frame, model):
    """
    Detectează vehicule într-un cadru folosind modelul YOLO.
    Args:
        frame (ndarray): Cadrul video pe care se face detecția.
        model: Modelul YOLO încărcat.
    Returns:
        detections (list): O listă cu detectiile sub forma [(x, y, w, h, label), ...]
    """
    results = model(frame)
    detections = []
    for result in results:
        if hasattr(result, 'boxes'):
            for box in result.boxes:
                cls = int(box.cls)
                confidence = float(box.conf[0])
                if cls in [2, 3, 5, 7]:  # Clasificăm doar vehicule (ex: mașini, camioane, etc.)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[cls]
                    if confidence < 0.5:  # Filtru pentru confidență scăzută
                        continue
                    label = f"{label} ({confidence:.2f})"
                    detections.append((x1, y1, x2 - x1, y2 - y1, label))
    return detections
