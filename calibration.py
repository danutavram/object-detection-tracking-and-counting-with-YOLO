# calibration.py
# Calibrarea unui videoclip: legatura dintre pixelii imaginii si metrii de pe drum.
#   - automata: geometria din marcajele discontinue (auto_calibration.py), iar tipul marcajului (dimensiunile lui
#     reale) identificat de program din raportul liniuta/pauza si din marimea aparenta a masinilor;
#   - manuala: patru puncte ale unui dreptunghi de pe drum, cu latimea si lungimea lui in metri.
# Calibrarea se salveaza langa videoclip (<videoclip>.calib.json) si se reincarca automat.
#
# Din linia de comanda:
#   python calibration.py video0.mp4                           (marcaj identificat automat)
#   python calibration.py video0.mp4 --marcaj 18 6 3.75        (perioada, lungime liniuta, latime banda - impuse)
import json
import os
import sys
import threading

import cv2
import numpy as np

import auto_calibration as ac

# Marcaje discontinue uzuale: (nume, perioada = liniuta + pauza, liniuta, latimea benzii), in metri.
# Valorile sunt orientative (din normativele nationale); programul alege tipul cel mai apropiat de ce masoara in imagine.
MARKING_STANDARDS = [
    ("Germania, autostrada (6 m + 12 m)", 18.0, 6.0, 3.75),
    ("Europa Centrala, extraurban (4 m + 8 m)", 12.0, 4.0, 3.5),
    ("Europa Centrala, urban (3 m + 6 m)", 9.0, 3.0, 3.25),
    ("Polonia, urban (2 m + 4 m)", 6.0, 2.0, 3.0),
    ("Marea Britanie, autostrada (2 m + 7 m)", 9.0, 2.0, 3.65),
    ("SUA (3 m + 9 m)", 12.19, 3.05, 3.66),
    ("Franta (3 m + 10 m)", 13.0, 3.0, 3.5),
    ("Italia, autostrada (4,5 m + 7,5 m)", 12.0, 4.5, 3.75),
    ("Spania, autostrada (5 m + 12 m)", 17.0, 5.0, 3.5),
    ("Spania, extraurban (3,5 m + 9 m)", 12.5, 3.5, 3.5),
    ("Spania, urban (2 m + 5,5 m)", 7.5, 2.0, 3.0),
    ("6 m + 9 m", 15.0, 6.0, 3.5),
]
AUTO_PRESET, CUSTOM_PRESET = "Automat (programul identifică marcajul)", "Personalizat"   # doar in interfata
ROAD_PRESETS = {AUTO_PRESET: None, **{name: dims for name, *dims in MARKING_STANDARDS}, CUSTOM_PRESET: None}

DUTY_TOLERANCE = 0.03                 # precizia cu care se masoara raportul liniuta / perioada
SCALE_SIGMA_LINES, SCALE_SIGMA_SINGLE = 0.3, 0.6   # incertitudinea (logaritmica) a scarii estimate din masini
PROVISIONAL_PERIOD, PROVISIONAL_LANE = 12.0, 3.65

MAX_METERS_PER_PIXEL = 0.5   # dincolo de aceasta rezolutie (zona indepartata) pozitiile nu mai sunt folosite


class Calibration:
    def __init__(self, H, method, info=None):
        self.H = np.asarray(H, dtype=float)      # omografia imagine (pixeli) -> drum (metri)
        self.method = method                      # "automata" / "manuala"
        self.info = info or {}

    def to_world(self, px, py):
        """Punct din imagine -> (X, Y) in metri pe planul drumului (Y = de-a lungul drumului)."""
        q = self.H @ np.array([px, py, 1.0])
        return q[0] / q[2], q[1] / q[2]

    def meters_per_pixel(self, px, py):
        """Cati metri de drum corespund unui pixel pe verticala, in punctul dat."""
        return abs(self.to_world(px, py)[1] - self.to_world(px, py - 1)[1])

    def usable(self, px, py):
        return self.meters_per_pixel(px, py) <= MAX_METERS_PER_PIXEL

    def describe(self):
        if self.method == "automata":
            text = f"Calibrare automata (eroare {self.info.get('rmse', 0):.2f} m)"
            if self.info.get("marking"):
                text += f"\nMarcaj: {self.info['marking']}"
                if self.info.get("confidence") == "scazuta":
                    text += "\nScara nesigura - viteze orientative"
            return text
        return "Calibrare manuala (4 puncte)"


def calibration_path(video_path):
    return os.path.splitext(video_path)[0] + ".calib.json"


def save_calibration(video_path, calib):
    data = {"method": calib.method, "H": calib.H.tolist(), "info": calib.info}
    with open(calibration_path(video_path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def load_calibration(video_path):
    """Calibrarea salvata pentru videoclip sau None daca nu exista."""
    path = calibration_path(video_path)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return Calibration(data["H"], data["method"], data.get("info"))


def detect_cars(frames):
    """Casetele autoturismelor din cadrele date (x1, y1, x2, y2): doar cele aflate in intregime in cadru si
    neacoperite de un vehicul din fata. Sunt folosite pentru estimarea scarii reale."""
    from model_loader import load_yolo_model
    from vehicle_detector import VEHICLE_CLASSES, CONFIDENCE_THRESHOLD
    model = load_yolo_model()
    cars = []
    for frame in frames:
        h, w = frame.shape[:2]
        result = model.predict(frame, classes=VEHICLE_CLASSES, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        boxes, classes = result.boxes.xyxy.tolist(), result.boxes.cls.int().tolist()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if classes[i] != 2 or x1 < 0.01 * w or x2 > 0.99 * w or y1 < 0.03 * h or y2 > 0.92 * h:
                continue
            # un vehicul aflat in fata (mai jos in imagine) care se suprapune ii ascunde baza: caseta nu este folosita
            if any(j != i and o[3] > y2 and not (o[2] < x1 - 5 or o[0] > x2 + 5 or o[1] > y2 + 5)
                   for j, o in enumerate(boxes)):
                continue
            cars.append((x1, y1, x2, y2))
    return np.array(cars, dtype=float).reshape(-1, 4)


# Imaginea drumului fara vehicule, pastrata in memorie pentru fiecare videoclip (cheie: cale + data modificarii):
# calculul ei dureaza ~15 s (citirea a 60 de cadre raspandite pe tot clipul + mediana) si e folosit de calibrare si de
# stabilizare la fiecare pornire a procesarii.
_BACKGROUNDS = {}
_BACKGROUND_LOCK = threading.Lock()


def _video_key(video_path):
    return os.path.abspath(video_path), os.path.getmtime(video_path)


def video_background(video_path):
    """Imaginea drumului fara vehicule (calculata o singura data pe sesiune pentru fiecare videoclip)."""
    with _BACKGROUND_LOCK:
        key = _video_key(video_path)
        if key not in _BACKGROUNDS:
            _BACKGROUNDS[key] = ac.background(video_path)
        return _BACKGROUNDS[key]


def analyze_video(video_path):
    """Citeste un esantion de cadre si intoarce (imaginea drumului fara vehicule, casetele autoturismelor)."""
    frames = ac.sample_frames(video_path)
    with _BACKGROUND_LOCK:
        key = _video_key(video_path)
        if key not in _BACKGROUNDS:
            _BACKGROUNDS[key] = ac.background(video_path, frames=frames)
        bg = _BACKGROUNDS[key]
    return bg, detect_cars(frames)


def identify_marking(duty, period_estimate, sigma):
    """
    Alege tipul de marcaj cel mai apropiat de masuratori: raportul liniuta/perioada (masurat direct pe marcaj, precis)
    si perioada estimata din marimea masinilor (aproximativa; poate lipsi).
    Returneaza (nume, perioada, liniuta, banda, incredere), cu increderea "ridicata" sau "scazuta".
    """
    scores = []
    for name, period, dash, lane in MARKING_STANDARDS:
        score = ((duty - dash / period) / DUTY_TOLERANCE) ** 2
        if period_estimate:
            score += (np.log(period / period_estimate) / sigma) ** 2
        scores.append((score, name, period, dash, lane))
    scores.sort()
    best, second = scores[0], scores[1]
    if best[0] > 9 and period_estimate:      # niciun marcaj cunoscut nu se potriveste: se foloseste direct estimarea
        return ("necunoscut, estimat din vehicule (%.1f m + %.1f m)" % (duty * period_estimate, (1 - duty) * period_estimate),
                float(period_estimate), float(duty * period_estimate), PROVISIONAL_LANE, "scazuta")
    confident = period_estimate and best[0] < 4 and second[0] - best[0] >= 1.0
    return best[1], best[2], best[3], best[4], "ridicata" if confident else "scazuta"


def _calibrate_lines(found, shape, vp, cars, marking):
    """Drum drept cu cel putin doua linii discontinue."""
    extra = {}
    if marking is None:
        duty = ac.measured_duty(found)
        first = ac.fit_lines(found, shape, vp, PROVISIONAL_PERIOD, PROVISIONAL_PERIOD * duty, PROVISIONAL_LANE)
        duty = ac.measured_duty(first["rays"])
        factor = ac.scale_from_cars(first["cam"]["H"], cars)
        estimate = PROVISIONAL_PERIOD * factor if factor else None
        name, period, dash, lane, confidence = identify_marking(duty, estimate, SCALE_SIGMA_LINES)
        extra = {"marking": name, "confidence": confidence, "duty_measured": duty, "period_from_cars": estimate}
        try:
            result = ac.fit_lines(found, shape, vp, period, dash, lane)
        except RuntimeError:               # liniuta tipului ales nu se potriveste: se pastreaza raportul masurat
            dash = period * duty
            result = ac.fit_lines(found, shape, vp, period, dash, lane)
    else:
        period, dash, lane = marking
        result = ac.fit_lines(found, shape, vp, period, dash, lane)
        extra = {"duty_measured": ac.measured_duty(result["rays"])}
    cam = result["cam"]
    info = {"period": period, "dash": dash, "lane": lane, "rmse": float(cam["rmse"]),
            "focal_px": float(cam["f"]), "camera_height_m": float(cam["h"]),
            "lines": len(result["rays"]), "points": int(sum(len(r["pts"]) for r in result["rays"])),
            "image_points": [np.round(r["pts"], 1).tolist() for r in result["rays"]], **extra}
    return Calibration(cam["H"], "automata", info)


def _calibrate_single_line(bg, cars, marking):
    """Drum cu o singura linie discontinua (poate fi si in curba). Scara este mai putin sigura decat cu mai multe linii."""
    fit = ac.calibrate_single_line(bg, cars)
    if marking is None:
        name, period, dash, lane, confidence = identify_marking(fit["duty"], fit["period_est"], SCALE_SIGMA_SINGLE)
        extra = {"marking": name, "confidence": "scazuta", "duty_measured": fit["duty"],
                 "period_from_cars": fit["period_est"]}
    else:
        (period, dash, lane), extra = marking, {"duty_measured": fit["duty"]}
    H = np.diag([period, period, 1.0]) @ fit["H"]         # din perioade in metri
    info = {"period": period, "dash": dash, "lane": lane, "rmse": fit["rmse"] * period, "focal_px": fit["f"],
            "camera_height_m": fit["h"] * period, "lines": 1, "points": len(fit["pts"]),
            "image_points": [np.round(fit["pts"], 1).tolist()], **extra}
    return Calibration(H, "automata", info)


def calibrate_auto(video_path, marking=None, bg=None, cars=None):
    """
    Calibrare automata din marcaje. marking = (perioada, liniuta, banda) in metri, sau None pentru ca tipul marcajului
    sa fie identificat de program. Ridica RuntimeError daca in imagine nu exista nicio linie discontinua utilizabila.
    """
    if bg is None:
        bg, cars = analyze_video(video_path)
    elif cars is None and marking is None:
        cars = detect_cars(ac.sample_frames(video_path))
    vp, found = ac.find_dashed_lines(bg)
    if len(found) >= 2:
        try:
            return _calibrate_lines(found, bg.shape[:2], vp, cars, marking)
        except RuntimeError:
            pass
    return _calibrate_single_line(bg, cars, marking)


def calibrate_manual(points, width_m, length_m):
    """
    Calibrare manuala din patru puncte ale unui dreptunghi de pe drum, date in ordinea:
    aproape-stanga, aproape-dreapta, departe-dreapta, departe-stanga.
    width_m = latimea dreptunghiului (de-a latul drumului), length_m = lungimea (de-a lungul drumului).
    """
    if width_m <= 0 or length_m <= 0:
        raise ValueError("Latimea si lungimea trebuie sa fie pozitive.")
    # punctele trebuie sa formeze un patrulater convex, parcurs pe contur (altfel laturile se incruciseaza
    # si transformarea rezultata nu are sens)
    turns = []
    for i in range(4):
        (ax, ay), (bx, by), (cx, cy) = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        turns.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx))
    if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        raise ValueError("Punctele nu formeaza un dreptunghi parcurs pe contur. Sterge punctele si marcheaza-le "
                         "in ordinea: aproape-stanga, aproape-dreapta, departe-dreapta, departe-stanga.")
    src = np.float32(points)
    dst = np.float32([[0, 0], [width_m, 0], [width_m, length_m], [0, length_m]])
    H = cv2.getPerspectiveTransform(src, dst)
    info = {"width": width_m, "length": length_m, "image_points": [np.round(src, 1).tolist()]}
    return Calibration(H, "manuala", info)


class Stabilizer:
    """Masoara deplasarea camerei fata de imaginea de referinta (fundalul); camera se misca lent, asa ca
    deplasarea se recalculeaza o data la cateva cadre."""

    def __init__(self, bg, every=5):
        self.ref = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        self.points = cv2.goodFeaturesToTrack(self.ref, 400, 0.01, 20)
        self.every, self.count, self.last = every, 0, (0.0, 0.0)

    def shift(self, frame):
        if self.count % self.every == 0:
            self.last = ac.global_shift(self.ref, self.points, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.count += 1
        return self.last


def draw_grid(image, calib, step=10.0):
    """Deseneaza peste imagine grila calibrarii (linii transversale din 'step' in 'step' metri) pentru verificare."""
    vis = image.copy()
    h, w = vis.shape[:2]
    Hinv = np.linalg.inv(calib.H)

    def to_image(X, Y):
        q = Hinv @ np.array([X, Y, 1.0])
        return int(round(q[0] / q[2])), int(round(q[1] / q[2]))

    groups = calib.info.get("image_points", [])
    for pts in groups:
        for px, py in pts:
            cv2.circle(vis, (int(px), int(py)), 5, (0, 0, 255), -1)
    world = [np.array([calib.to_world(px, py) for px, py in pts]) for pts in groups if len(pts)]
    if not world:
        return vis
    allw = np.vstack(world)
    margin = calib.info.get("lane", 0.0)
    x0, x1 = allw[:, 0].min() - margin, allw[:, 0].max() + margin
    y0 = np.floor(allw[:, 1].min() / step) * step
    y1 = np.ceil(allw[:, 1].max() / step) * step
    for Y in np.arange(y0, y1 + 0.1, step):
        cv2.line(vis, to_image(x0, Y), to_image(x1, Y), (0, 255, 255), 2)
        tx, ty = to_image(x1, Y)
        cv2.putText(vis, f"{Y - y0:.0f} m", (min(tx + 8, w - 90), ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    xs = sorted({round(float(np.median(wp[:, 0])), 2) for wp in world}) if calib.method == "automata" else []
    for X in [x0, *xs, x1]:
        cv2.line(vis, to_image(X, y0), to_image(X, y1), (0, 255, 0), 2)
    return vis


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Utilizare: python calibration.py <videoclip> [--marcaj PERIOADA LINIUTA BANDA]")
    video = sys.argv[1]
    dims = tuple(float(v) for v in sys.argv[3:6]) if "--marcaj" in sys.argv else None
    background, vehicles = analyze_video(video)
    try:
        c = calibrate_auto(video, dims, bg=background, cars=vehicles)
    except RuntimeError as err:
        sys.exit(f"{err}\nFolositi calibrarea manuala din interfata (python gui.py).")
    save_calibration(video, c)
    preview = os.path.splitext(video)[0] + ".calib.jpg"
    cv2.imwrite(preview, draw_grid(background, c))
    print(c.describe())
    print(c.info["lines"], "linii discontinue,", c.info["points"], "puncte;",
          f"perioada marcajului {c.info['period']:g} m; camera la {c.info['camera_height_m']:.1f} m inaltime")
    if c.info.get("period_from_cars"):
        print(f"Perioada estimata independent, din marimea masinilor: {c.info['period_from_cars']:.1f} m")
    print("Salvat:", calibration_path(video), "| verificare vizuala:", preview)
