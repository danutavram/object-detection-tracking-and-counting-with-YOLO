# main_tracker.py

import cv2
from model_loader import load_yolo_model
from video_manager import open_video, initialize_video_writer, release_video
from vehicle_detector import detect_vehicles
from tracker import Tracker
from red_vehicle_counter import is_vehicle_red


def real_time_detection(video_path, output_path):
    # Incarca modelul YOLO
    model = load_yolo_model("yolo11s.pt")

    # Deschide videoclipul
    cap = open_video(video_path)
    if cap is None:
        return

    # Proprietati video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initializeaza scrierea videoclipului de iesire
    out = initialize_video_writer(frame_width, frame_height, output_path, fps)

    # Creeaza tracker
    tracker = Tracker()
    counted_red_vehicles = set()  # Vehicule rosii contorizate pentru linia rosie
    red_vehicle_count = 0  # Contor vehicule rosii
    entry_count, exit_count = 0, 0  # Contoare vehicule intrate si iesite

    # Definirea liniilor
    detection_line = int(frame_height * 0.5)  # Linie galbena (intrare/iesire)
    red_detection_line = int(frame_height * 0.6)  # Linie rosie pentru vehicule rosii

    # Procesare cadre video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detecteaza vehicule
        detections = detect_vehicles(frame, model)
        tracked_objects, entered, exited, speeds = tracker.update(
            [(x, y, w, h) for x, y, w, h, _ in detections], detection_line, fps)

        entry_count += entered  # Actualizam numarul vehiculelor intrate
        exit_count += exited    # Actualizam numarul vehiculelor iesite

        # Procesare detectari
        for obj, det in zip(tracked_objects, detections):
            x, y, w, h, obj_id = obj
            label = det[-1]
            speed = speeds.get(obj_id, 0)
            cx, cy = x + w // 2, y + h // 2  # Calculam centrul

            # ROI vehicul
            x_end, y_end = min(x + w, frame_width), min(y + h, frame_height)
            x_start, y_start = max(x, 0), max(y, 0)
            vehicle_roi = frame[y_start:y_end, x_start:x_end]
            
            # Detecteaza vehicule rosii care trec linia rosie
            if abs(cy - red_detection_line) <= 10:
                if is_vehicle_red(vehicle_roi):
                    if obj_id not in counted_red_vehicles:
                        counted_red_vehicles.add(obj_id)
                        red_vehicle_count += 1
                        # Evidentiaza vehiculul rosu
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Red Vehicle", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Evidentiaza toate vehiculele detectate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"Viteza: {speed:.1f} px/s", (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Deseneaza liniile orizontale
        cv2.line(frame, (0, detection_line), (frame_width, detection_line), (0, 255, 255), 2)  # Linie galbena
        cv2.putText(frame, f"Vehicule Intrate: {entry_count}", (10, detection_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Vehicule Iesite: {exit_count}", (10, detection_line + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.line(frame, (0, red_detection_line), (frame_width, red_detection_line), (0, 0, 255), 2)  # Linie rosie
        cv2.putText(frame, f"Vehicule Rosii: {red_vehicle_count}", (10, red_detection_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Scrie cadrul in fisierul de iesire
        out.write(frame)

    # Elibereaza resurse
    release_video(cap, out)