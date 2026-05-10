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
    counted_red_vehicles = set()
    red_vehicle_count = 0
    entry_count, exit_count = 0, 0

    # Definirea liniilor
    detection_line = int(frame_height * 0.5)      # Linie galbena (intrare/iesire)
    speed_line_1 = int(frame_height * 0.35)        # Linia de viteza 1 (sus) - cyan
    speed_line_2 = int(frame_height * 0.65)        # Linia de viteza 2 (jos) - cyan
    red_detection_line = int(frame_height * 0.70)  # Linie rosie pentru vehicule rosii

    frame_number = 0

    # Procesare cadre video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detecteaza vehicule
        detections = detect_vehicles(frame, model)
        tracked_objects, entered, exited, speeds = tracker.update(
            [(x, y, w, h) for x, y, w, h, _ in detections],
            detection_line, fps, frame_number, speed_line_1, speed_line_2)

        entry_count += entered
        exit_count += exited

        # Procesare detectari
        for obj, det in zip(tracked_objects, detections):
            x, y, w, h, obj_id = obj
            label = det[-1]
            speed = speeds.get(obj_id, 0)
            cx, cy = x + w // 2, y + h // 2

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
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Red Vehicle", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Evidentiaza toate vehiculele detectate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Afiseaza viteza doar daca a fost calculata (vehiculul a traversat ambele linii)
            if speed > 5:
                cv2.putText(frame, f"{speed:.0f} km/h", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # --- Deseneaza liniile ---
        # Linii de viteza (cyan, punctate)
        for lx in range(0, frame_width, 20):
            cv2.line(frame, (lx, speed_line_1), (lx + 10, speed_line_1), (255, 255, 0), 2)
            cv2.line(frame, (lx, speed_line_2), (lx + 10, speed_line_2), (255, 255, 0), 2)
        cv2.putText(frame, "SPEED LINE 1", (frame_width - 250, speed_line_1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "SPEED LINE 2", (frame_width - 250, speed_line_2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Linie galbena (contorizare)
        cv2.line(frame, (0, detection_line), (frame_width, detection_line), (0, 255, 255), 2)
        cv2.putText(frame, f"Vehicule Intrate: {entry_count}", (10, detection_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Vehicule Iesite: {exit_count}", (10, detection_line + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # Linie rosie (vehicule rosii)
        cv2.line(frame, (0, red_detection_line), (frame_width, red_detection_line), (0, 0, 255), 2)
        cv2.putText(frame, f"Vehicule Rosii: {red_vehicle_count}", (10, red_detection_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Scrie cadrul in fisierul de iesire
        out.write(frame)
        frame_number += 1

    # Elibereaza resurse
    release_video(cap, out)


def real_time_detection_with_callback(video_path, output_path, frame_callback=None, stop_event=None):
    """Versiune cu callback pentru integrarea cu GUI."""
    model = load_yolo_model("yolo11s.pt")

    cap = open_video(video_path)
    if cap is None:
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = initialize_video_writer(frame_width, frame_height, output_path, fps)
    tracker = Tracker()
    counted_red_vehicles = set()
    red_vehicle_count = 0
    entry_count, exit_count = 0, 0
    max_speed = 0.0

    detection_line = int(frame_height * 0.5)
    speed_line_1 = int(frame_height * 0.35)
    speed_line_2 = int(frame_height * 0.65)
    red_detection_line = int(frame_height * 0.70)

    frame_number = 0

    while cap.isOpened():
        if stop_event and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_vehicles(frame, model)
        tracked_objects, entered, exited, speeds = tracker.update(
            [(x, y, w, h) for x, y, w, h, _ in detections],
            detection_line, fps, frame_number, speed_line_1, speed_line_2)

        entry_count += entered
        exit_count += exited

        if speeds:
            current_max = max(speeds.values())
            if current_max > max_speed:
                max_speed = current_max

        for obj, det in zip(tracked_objects, detections):
            x, y, w, h, obj_id = obj
            label = det[-1]
            speed = speeds.get(obj_id, 0)
            cx, cy = x + w // 2, y + h // 2

            x_end = min(x + w, frame_width)
            y_end = min(y + h, frame_height)
            x_start = max(x, 0)
            y_start = max(y, 0)
            vehicle_roi = frame[y_start:y_end, x_start:x_end]

            if abs(cy - red_detection_line) <= 10:
                if is_vehicle_red(vehicle_roi):
                    if obj_id not in counted_red_vehicles:
                        counted_red_vehicles.add(obj_id)
                        red_vehicle_count += 1
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Red Vehicle", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if speed > 5:
                cv2.putText(frame, f"{speed:.0f} km/h", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for lx in range(0, frame_width, 20):
            cv2.line(frame, (lx, speed_line_1), (lx + 10, speed_line_1), (255, 255, 0), 2)
            cv2.line(frame, (lx, speed_line_2), (lx + 10, speed_line_2), (255, 255, 0), 2)
        cv2.putText(frame, "SPEED LINE 1", (frame_width - 250, speed_line_1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "SPEED LINE 2", (frame_width - 250, speed_line_2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.line(frame, (0, detection_line), (frame_width, detection_line), (0, 255, 255), 2)
        cv2.putText(frame, f"Vehicule Intrate: {entry_count}", (10, detection_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Vehicule Iesite: {exit_count}", (10, detection_line + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.line(frame, (0, red_detection_line), (frame_width, red_detection_line), (0, 0, 255), 2)
        cv2.putText(frame, f"Vehicule Rosii: {red_vehicle_count}", (10, red_detection_line - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)

        if frame_callback:
            stats = {
                'entry_count': entry_count,
                'exit_count': exit_count,
                'red_vehicle_count': red_vehicle_count,
                'frame_number': frame_number,
                'total_frames': total_frames,
                'max_speed': max_speed,
            }
            frame_callback(frame.copy(), stats)

        frame_number += 1

    release_video(cap, out)
