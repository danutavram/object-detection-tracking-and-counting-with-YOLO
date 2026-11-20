# timp_detectie.py
# Timpul de detectie pe cadru (doar reteaua YOLO, fara urmarire si fara restul programului) pentru mai multe modele,
# masurat pe aceleasi 20 de cadre din video0.mp4 (o trecere de incalzire, apoi mediana a 5 treceri), unul dupa altul.
#
# Utilizare (din folderul proiectului):
#   python experimente/timp_detectie.py yolo11n yolo11s yolo11m yolo11l yolo11x yolo26n yolo26s yolo26m yolo26l yolo26x
#   python experimente/timp_detectie.py --device cpu yolo26s     (implicit: placa video, daca PyTorch o vede)
# Ponderile lipsa se descarca automat. Rezultatele se adauga in experimente/timp_detectie.json, cu cheia
# "<model> (<dispozitiv>)", deci timpii pe procesor si pe placa video se pastreaza separat.
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
os.chdir(PROJECT)

from ultralytics import YOLO
from vehicle_detector import CONFIDENCE_THRESHOLD, VEHICLE_CLASSES

args = sys.argv[1:]
device = None                                   # None = alegerea Ultralytics (placa video, daca exista)
if "--device" in args:
    i = args.index("--device")
    device = args[i + 1]
    del args[i:i + 2]

cap = cv2.VideoCapture("video0.mp4")
frames = []
for i in np.linspace(100, 1000, 20).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, frame = cap.read()
    if ok:
        frames.append(frame)

OUT = os.path.join(HERE, "timp_detectie.json")
results = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
for name in args:
    model = YOLO(name + ".pt")
    opts = dict(classes=VEHICLE_CLASSES, conf=CONFIDENCE_THRESHOLD, verbose=False, device=device)
    model.predict(frames[0], **opts)                    # creeaza predictorul (si afla dispozitivul)
    dev = str(model.predictor.device)       # copia modelului folosita la predictie (model.model ramane pe CPU)
    cuda = dev.startswith("cuda")
    # O trecere completa de incalzire (pe placa video o trecere dureaza sub 0,5 s si placa abia isi urca frecventa),
    # apoi 5 treceri si mediana lor.
    passes = 5
    for frame in frames:
        model.predict(frame, **opts)
    per_pass, boxes = [], 0
    for _ in range(passes):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for frame in frames:
            boxes += len(model.predict(frame, **opts)[0].boxes)
        if cuda:
            torch.cuda.synchronize()    # placa video lucreaza asincron: se asteapta terminarea ultimului cadru
        per_pass.append(1000 * (time.perf_counter() - start) / len(frames))
    ms = float(np.median(per_pass))
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    gpu = torch.cuda.get_device_name(0) if cuda else None
    results[f"{name} ({dev})"] = dict(ms=ms, ms_per_pass=per_pass, vehicles_per_frame=boxes / (passes * len(frames)),
                                      params_M=params, device=dev, gpu=gpu, torch=torch.__version__)
    print(f"{name:9s} {ms:7.1f} ms/cadru  {boxes / (passes * len(frames)):5.1f} vehicule/cadru  {params:6.1f} M parametri"
          f"  ({dev})", flush=True)
json.dump(results, open(OUT, "w", encoding="utf-8"), indent=1)
