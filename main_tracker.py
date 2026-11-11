# main_tracker.py

import csv
import os

import cv2
from model_loader import load_yolo_model
from video_manager import open_video, initialize_video_writer, release_video
from vehicle_detector import detect_vehicles
from tracker import Tracker
from calibration import load_calibration, Stabilizer
from auto_calibration import background


def draw_annotations(frame, detections, tracker, stats):
    """Deseneaza casetele, etichetele (ID, tip, viteza), linia de numarare si panoul cu totaluri."""
    frame_width = frame.shape[1]
    for x, y, w, h, tid, cls, confidence, label in detections:
        color = (0, 0, 255) if tid in tracker.red_ids else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"#{tid} {label.split(' ')[0]}"
        if tid in tracker.speeds:
            text += f" {tracker.speeds[tid]:.0f} km/h"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), (0, 0, 0), -1)
        cv2.putText(frame, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255) if tid in tracker.speeds else (255, 255, 255), 2)

    line = tracker.counting_line
    cv2.line(frame, (0, line), (frame_width, line), (0, 255, 255), 2)

    # panoul cu totaluri, pe fundal inchis, in coltul din stanga sus
    speed_text = f"{stats['max_speed']:.0f} km/h" if tracker.calibration is not None else "necalibrat"
    texts = [(f"Se apropie: {stats['entry_count']}", (255, 255, 255)),
             (f"Se departeaza: {stats['exit_count']}", (255, 255, 255)),
             (f"Vehicule rosii: {stats['red_vehicle_count']}", (80, 80, 255)),
             (f"Viteza maxima: {speed_text}", (0, 255, 255))]
    cv2.rectangle(frame, (10, 10), (400, 20 + 35 * len(texts)), (0, 0, 0), -1)
    for i, (text, color) in enumerate(texts):
        cv2.putText(frame, text, (20, 45 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def save_vehicle_table(path, rows):
    """Salveaza tabelul vehiculelor numarate in format CSV (se deschide direct in Excel)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["ID", "Timp (s)", "Sens", "Tip", "Rosu", "Viteza (km/h)"])
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def real_time_detection_with_callback(video_path, output_path, frame_callback=None, stop_event=None):
    """
    Proceseaza videoclipul: detectie + urmarire, numarare, vehicule rosii, estimarea vitezei.
    Viteza se masoara doar daca exista o calibrare salvata pentru videoclip (vezi calibration.py / interfata).

    Args:
        video_path (str): Calea catre videoclipul de intrare.
        output_path (str): Calea catre videoclipul adnotat de iesire (langa el se salveaza si tabelul CSV).
        frame_callback: Functie apelata dupa fiecare cadru: frame_callback(frame, stats) (folosita de GUI).
        stop_event: threading.Event; daca este setat, procesarea se opreste.

    Returns:
        dict cu statisticile finale (sau None daca videoclipul nu poate fi deschis).
    """
    model = load_yolo_model()

    calibration = load_calibration(video_path)
    stabilizer = Stabilizer(background(video_path)) if calibration is not None else None

    cap = open_video(video_path)
    if cap is None:
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)                       # valoare reala (float), fara trunchiere
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = initialize_video_writer(frame_width, frame_height, output_path, fps)
    tracker = Tracker(fps, frame_width, frame_height, calibration)

    frame_number = 0
    stats = {'entry_count': 0, 'exit_count': 0, 'red_vehicle_count': 0, 'max_speed': 0.0}

    while cap.isOpened():
        if stop_event is not None and stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_vehicles(frame, model)
        shift = stabilizer.shift(frame) if stabilizer is not None else (0.0, 0.0)
        stats = tracker.update(frame, detections, frame_number, shift)
        draw_annotations(frame, detections, tracker, stats)

        out.write(frame)
        if frame_callback:
            frame_callback(frame.copy(), dict(stats, frame_number=frame_number, total_frames=total_frames,
                                              calibrated=calibration is not None))
        frame_number += 1

    release_video(cap, out)
    table_path = os.path.splitext(output_path)[0] + ".csv"
    save_vehicle_table(table_path, tracker.vehicles_table(model.names))
    return dict(stats, frames=frame_number, calibrated=calibration is not None, table_path=table_path,
                speeds=dict(tracker.speeds), counted=dict(tracker.counted),
                cross_frame=dict(tracker.cross_frame), red_ids=sorted(tracker.red_ids))


def real_time_detection(video_path, output_path):
    """Varianta fara interfata grafica: proceseaza videoclipul si afiseaza rezultatele in consola."""
    result = real_time_detection_with_callback(video_path, output_path)
    if result is not None:
        print(f"Se apropie:     {result['entry_count']}")
        print(f"Se departeaza:  {result['exit_count']}")
        print(f"Vehicule rosii: {result['red_vehicle_count']}")
        if result['calibrated']:
            print(f"Viteza maxima:  {result['max_speed']:.0f} km/h")
        else:
            print("Viteza:         videoclip necalibrat (rulati calibrarea din interfata sau calibration.py)")
        print(f"Videoclip salvat: {output_path}")
        print(f"Tabel vehicule:   {result['table_path']}")
    return result
