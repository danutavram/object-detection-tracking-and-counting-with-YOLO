# main_tracker.py

import csv
import os

import cv2
from model_loader import load_yolo_model
from video_manager import open_video, initialize_video_writer, release_video
from vehicle_detector import detect_vehicles, CLASS_NAMES
from tracker import Tracker
from calibration import load_calibration, video_background, Stabilizer


def create_plate_reader():
    """
    (cititorul de numere de inmatriculare, None) sau (None, motivul) daca nu poate fi pornit: biblioteca fast-alpr
    lipseste sau, la prima rulare pe un calculator, modelele ei nu se pot descarca (fara internet).
    Restul programului merge si fara el.
    """
    try:
        from plate_reader import PlateReader
        return PlateReader(), None
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"
        print(f"Citirea numerelor de inmatriculare este dezactivata: {reason}")
        return None, reason


def draw_annotations(frame, detections, tracker, stats, plates):
    """Deseneaza casetele, etichetele (ID, tip, viteza, numar), linia de numarare si panoul cu totaluri."""
    frame_width = frame.shape[1]
    for x, y, w, h, tid, cls, confidence, label in detections:
        color = (0, 0, 255) if tid in tracker.red_ids else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"#{tid} {CLASS_NAMES[cls]}"
        if tid in tracker.speeds:
            text += f" {tracker.speeds[tid]:.0f} km/h"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), (0, 0, 0), -1)
        cv2.putText(frame, text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255) if tid in tracker.speeds else (255, 255, 255), 2)
        if tid in plates:        # numarul de inmatriculare: sub caseta, cu litere mari (lizibil si in previzualizare)
            (pw, ph), _ = cv2.getTextSize(plates[tid], cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.rectangle(frame, (x, y + h), (x + pw + 10, y + h + ph + 14), (0, 0, 0), -1)
            cv2.putText(frame, plates[tid], (x + 5, y + h + ph + 7), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 3)

    line = tracker.counting_line
    cv2.line(frame, (0, line), (frame_width, line), (0, 255, 255), 2)

    # panoul cu totaluri, pe fundal inchis, in coltul din stanga sus
    speed_text = f"{stats['max_speed']:.0f} km/h" if tracker.calibration is not None else "necalibrat"
    texts = [(f"Se apropie: {stats['entry_count']}", (255, 255, 255)),
             (f"Se departeaza: {stats['exit_count']}", (255, 255, 255)),
             (f"Vehicule rosii: {stats['red_vehicle_count']}", (80, 80, 255)),
             (f"Viteza maxima: {speed_text}", (0, 255, 255))]
    if plates is not None and stats.get('plates_read'):
        texts.append((f"Numere citite: {stats['plates_read']}", (255, 200, 0)))
    cv2.rectangle(frame, (10, 10), (400, 20 + 35 * len(texts)), (0, 0, 0), -1)
    for i, (text, color) in enumerate(texts):
        cv2.putText(frame, text, (20, 45 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def save_vehicle_table(path, rows):
    """Salveaza tabelul vehiculelor numarate in format CSV (se deschide direct in Excel)."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["ID", "Timp (s)", "Sens", "Tip", "Rosu", "Viteza (km/h)", "Numar inmatriculare"])
        for row in rows:
            writer.writerow(["" if v is None else v for v in row])


def real_time_detection_with_callback(video_path, output_path, frame_callback=None, stop_event=None,
                                      read_plates=True):
    """
    Proceseaza videoclipul: detectie + urmarire, numarare, vehicule rosii, estimarea vitezei, citirea numerelor
    de inmatriculare (read_plates; are efect doar daca placutele sunt suficient de mari in imagine).
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
    plate_reader, plates_problem = create_plate_reader() if read_plates else (None, None)
    # starea citirii numerelor, pentru interfata: None = activa
    plates_status = None if plate_reader is not None else ("oprita" if not read_plates else f"indisponibila ({plates_problem})")

    calibration = load_calibration(video_path)
    stabilizer = Stabilizer(video_background(video_path)) if calibration is not None else None

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
    stats = {'entry_count': 0, 'exit_count': 0, 'red_vehicle_count': 0, 'max_speed': 0.0, 'plates_read': 0}

    while cap.isOpened():
        if stop_event is not None and stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_vehicles(frame, model)
        shift = stabilizer.shift(frame) if stabilizer is not None else (0.0, 0.0)
        stats = tracker.update(frame, detections, frame_number, shift)
        plates = plate_reader.update(frame, detections, frame_number) if plate_reader is not None else {}
        # numerele raportate (imagine, interfata, CSV) sunt doar ale vehiculelor numarate
        counted_plates = [(tid, plates[tid], tracker.speeds.get(tid))
                          for tid in sorted(tracker.counted, key=lambda i: tracker.cross_frame[i]) if tid in plates]
        stats['plates_read'] = len(counted_plates)
        draw_annotations(frame, detections, tracker, stats, plates)

        out.write(frame)
        if frame_callback:
            frame_callback(frame.copy(), dict(stats, frame_number=frame_number, total_frames=total_frames,
                                              calibrated=calibration is not None, plates=counted_plates,
                                              plates_status=plates_status,
                                              line_ratio=tracker.counting_line / frame_height))
        frame_number += 1

    release_video(cap, out)
    table_path = os.path.splitext(output_path)[0] + ".csv"
    plates = plate_reader.plates if plate_reader is not None else {}
    save_vehicle_table(table_path, tracker.vehicles_table(CLASS_NAMES, plates))
    return dict(stats, frames=frame_number, calibrated=calibration is not None, table_path=table_path,
                speeds=dict(tracker.speeds), counted=dict(tracker.counted), plates=dict(plates),
                cross_frame=dict(tracker.cross_frame), red_ids=sorted(tracker.red_ids),
                plates_status=plates_status, line_ratio=tracker.counting_line / frame_height)


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
        print(f"Numere citite:  {result['plates_read']}" if result['plates_status'] is None
              else f"Numere citite:  citirea este {result['plates_status']}")
        print(f"Linia de numarare: {100 * result['line_ratio']:.0f} % din inaltimea imaginii (aleasa automat)")
        print(f"Videoclip salvat: {output_path}")
        print(f"Tabel vehicule:   {result['table_path']}")
    return result
