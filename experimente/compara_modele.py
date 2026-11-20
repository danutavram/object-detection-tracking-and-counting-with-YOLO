# compara_modele.py
# Compara variantele YOLO (si pragul de incredere / configuratia ByteTrack) pe tot lantul de prelucrare al programului,
# fara citirea numerelor de inmatriculare, cu calibrarea salvata a fiecarui videoclip (<video>.calib.json).
# Pe video0.mp4 numararea se compara cu adevarul de referinta (ground_truth.json), ca in test_sistem.py.
#
# Utilizare (din folderul proiectului):
#   python experimente/compara_modele.py yolo11s@video0 yolo26s@video0 yolo11s@video_test2
#   python experimente/compara_modele.py yolo26s@video0@0.25           (prag de incredere 0,25 in loc de 0,5)
#   python experimente/compara_modele.py yolo11s@video0@0.1@bytetrack_strict.yaml
# Forma unei rulari: model@videoclip[@prag[@configuratie_tracker.yaml]]. Ponderile lipsa se descarca automat.
# Rezultatele se adauga in experimente/rezultate_modele.json; o rulare existenta deja in fisier nu se repeta.
# Timpul pe cadru se masoara intre primul si ultimul cadru (fara incarcarea modelului si calculul fundalului).
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from ultralytics import YOLO

import main_tracker
import model_loader
import vehicle_detector
from test_sistem import match

OUT = os.path.join(HERE, "rezultate_modele.json")


def run(spec, gt):
    parts = spec.split("@")
    name, video = parts[0], parts[1]
    conf = float(parts[2]) if len(parts) > 2 else 0.5
    tracker_cfg = os.path.join(HERE, parts[3]) if len(parts) > 3 else "bytetrack.yaml"

    # modelul, pragul si configuratia urmaritorului se schimba fara a modifica modulele programului
    loader = lambda model_path=None: YOLO(name + ".pt")
    model_loader.load_yolo_model = loader
    main_tracker.load_yolo_model = loader
    vehicle_detector.CONFIDENCE_THRESHOLD = conf
    vehicle_detector.TRACKER_CONFIG = tracker_cfg

    stamps = []
    out_video = os.path.join(HERE, "_tmp_%s.mp4" % spec.replace("@", "_"))
    r = main_tracker.real_time_detection_with_callback(video + ".mp4", out_video, read_plates=False,
                                                       frame_callback=lambda f, s: stamps.append(time.perf_counter()))
    with open(r["table_path"], encoding="utf-8-sig") as f:
        table = list(csv.reader(f, delimiter=";"))[1:]
    for p in (out_video, r["table_path"]):
        if os.path.exists(p):
            os.remove(p)
    rec = dict(model=name, video=video, conf=conf, tracker=os.path.basename(tracker_cfg), frames=r["frames"],
               ms_per_frame=1000 * (stamps[-1] - stamps[0]) / (len(stamps) - 1),
               entry=r["entry_count"], exit=r["exit_count"], red=r["red_vehicle_count"],
               counted=len(r["counted"]), with_speed=sum(1 for i in r["counted"] if i in r["speeds"]), table=table)
    if video == "video0":
        for key, direction in (("L", "in"), ("R", "out")):
            frames = [r["cross_frame"][i] for i, d in r["counted"].items() if d == direction]
            tp, extra, missed = match(frames, gt[key])
            rec["gt_" + key] = dict(tp=tp, false=len(extra), missed=len(missed), false_frames=extra)
    return rec


def main():
    gt = json.load(open("ground_truth.json", encoding="utf-8"))
    results = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for spec in sys.argv[1:]:
        if spec in results:
            continue
        rec = run(spec, gt)
        results[spec] = rec
        json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1)
        line = (f"{spec:45s} {rec['ms_per_frame']:5.0f} ms/cadru | numarate {rec['entry']:3d} + {rec['exit']:3d}"
                f" | rosii {rec['red']:2d} | cu viteza {rec['with_speed']}/{rec['counted']}")
        if "gt_L" in rec:
            line += (" | referinta: stang %d/34 (+%d false), drept %d/34 (+%d false)"
                     % (rec["gt_L"]["tp"], rec["gt_L"]["false"], rec["gt_R"]["tp"], rec["gt_R"]["false"]))
        print(line, flush=True)


if __name__ == "__main__":
    main()
