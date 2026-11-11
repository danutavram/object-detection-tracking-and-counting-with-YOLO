# vehicle_detector.py

VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: 2 - car, 3 - motorcycle, 5 - bus, 7 - truck
CONFIDENCE_THRESHOLD = 0.5
TRACKER_CONFIG = "bytetrack.yaml"


def detect_vehicles(frame, model):
    """
    Detecteaza si urmareste vehiculele dintr-un cadru folosind modelul YOLO si algoritmul ByteTrack.

    Args:
        frame (ndarray): Cadrul video curent.
        model: Modelul YOLO incarcat.

    Returns:
        detections (list): Lista de vehicule sub forma
            [(x, y, w, h, track_id, cls, confidence, label), ...]
    """
    result = model.track(frame, persist=True, tracker=TRACKER_CONFIG, classes=VEHICLE_CLASSES,
                         conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    detections = []
    if result.boxes.id is None:   # niciun vehicul urmarit in acest cadru
        return detections

    boxes = result.boxes
    for (x1, y1, x2, y2), track_id, cls, confidence in zip(boxes.xyxy.int().tolist(), boxes.id.int().tolist(),
                                                           boxes.cls.int().tolist(), boxes.conf.tolist()):
        label = f"{model.names[cls]} ({confidence:.2f})"
        detections.append((x1, y1, x2 - x1, y2 - y1, track_id, cls, confidence, label))
    return detections
