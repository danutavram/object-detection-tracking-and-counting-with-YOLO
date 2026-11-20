# vehicle_detector.py

# COCO: 2 - car, 3 - motorcycle, 5 - bus, 6 - train, 7 - truck. In clipurile de test nu exista niciun tren, dar modelul
# confunda uneori un vehicul mare de pe drum cu un tren (pe video_test2, autoutilitara de la 25,07 s era eliminata si
# nenumarata); de aceea clasa e inclusa si afisata ca „vehicul mare”. Pe celelalte clipuri includerea ei nu schimba nimic.
VEHICLE_CLASSES = [2, 3, 5, 6, 7]
# Denumirile afisate in imagine si in tabel (fara diacritice: fontul OpenCV nu le are).
CLASS_NAMES = {2: "autoturism", 3: "motocicleta", 5: "autobuz", 6: "vehicul mare", 7: "camion"}
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
        label = f"{CLASS_NAMES[cls]} ({confidence:.2f})"
        detections.append((x1, y1, x2 - x1, y2 - y1, track_id, cls, confidence, label))
    return detections
