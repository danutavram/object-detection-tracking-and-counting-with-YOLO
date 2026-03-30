# vehicle_detector.py

def detect_vehicles(frame, model):
    """
    Detecteaza vehicule intr-un cadru folosind modelul YOLO.
    Args:
        frame (ndarray): Cadru video pe care se face detectia.
        model: Modelul YOLO incarcat.
    Returns:
        detections (list): O lista cu detectiile sub forma [(x, y, w, h, label), ...]
    """
    # Aplicam modelul YOLO pentru a detecta obiecte in cadru
    results = model(frame)
    detections = []  # Lista pentru a stoca detectiile

    for result in results:
        if hasattr(result, 'boxes'):  # Verificam daca exista cutii de delimitare
            for box in result.boxes:
                cls = int(box.cls)  # Clasa obiectului detectat (ex: masina, camion, etc.)
                confidence = float(box.conf[0])  # Confidenta detectiei

                # Clasificam doar obiectele care sunt vehicule (cls in [2, 3, 5, 7])
                if cls in [2, 3, 5, 7]:
                    # Coordonatele colturilor cutiei de delimitare
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Eticheta obiectului (ex: masina, camion) si confidenta
                    label = model.names[cls]
                    
                    # Filtru pentru a elimina detectiile cu o confidenta scazuta
                    if confidence < 0.5:
                        continue
                    
                    # Adaugam eticheta cu confidenta in formatul "label"
                    label = f"{label} ({confidence:.2f})"
                    # Adaugam detectia sub forma (x1, y1, latime, inaltime, eticheta)
                    detections.append((x1, y1, x2 - x1, y2 - y1, label))
    
    return detections