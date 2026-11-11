# calibration.py
# Calibrarea unui videoclip: legatura dintre pixelii imaginii si metrii de pe drum.
#   - automata: din marcajele discontinue (auto_calibration.py) + dimensiunile reale ale marcajului;
#   - manuala: patru puncte ale unui dreptunghi de pe drum, cu latimea si lungimea lui in metri.
# Calibrarea se salveaza langa videoclip (<videoclip>.calib.json) si se reincarca automat.
#
# Din linia de comanda:
#   python calibration.py video0.mp4 --marcaj 18 6 3.75        (perioada, lungime liniuta, latime banda)
import json
import os
import sys

import cv2
import numpy as np

import auto_calibration as ac

# Dimensiunile marcajului discontinuu dintre benzi: (perioada = liniuta + pauza, liniuta, latimea benzii), in metri.
ROAD_PRESETS = {
    "Autostrada Germania (6 m + 12 m, banda 3,75 m)": (18.0, 6.0, 3.75),
    "Autostrada Marea Britanie (2 m + 7 m, banda 3,65 m)": (9.0, 2.0, 3.65),
    "Personalizat": None,
}

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
            return f"Calibrare automata (eroare {self.info.get('rmse', 0):.2f} m)"
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


def calibrate_auto(video_path, period, dash, lane, bg=None):
    """Calibrare automata din marcaje. Ridica RuntimeError daca marcajele nu pot fi folosite."""
    if bg is None:
        bg = ac.background(video_path)
    result = ac.calibrate(bg, period, dash, lane)
    cam = result["cam"]
    info = {"period": period, "dash": dash, "lane": lane, "rmse": float(cam["rmse"]),
            "focal_px": float(cam["f"]), "camera_height_m": float(cam["h"]),
            "lines": len(result["rays"]), "points": int(sum(len(r["pts"]) for r in result["rays"])),
            "image_points": [np.round(r["pts"], 1).tolist() for r in result["rays"]]}
    return Calibration(cam["H"], "automata", info)


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
    dims = tuple(float(v) for v in sys.argv[3:6]) if "--marcaj" in sys.argv else (18.0, 6.0, 3.75)
    background = ac.background(video)
    try:
        c = calibrate_auto(video, *dims, bg=background)
    except RuntimeError as err:
        sys.exit(f"{err}\nFolositi calibrarea manuala din interfata (python gui.py).")
    save_calibration(video, c)
    preview = os.path.splitext(video)[0] + ".calib.jpg"
    cv2.imwrite(preview, draw_grid(background, c))
    print(c.describe(), "-", c.info["lines"], "linii discontinue,", c.info["points"], "puncte;",
          f"camera la {c.info['camera_height_m']:.1f} m inaltime")
    print("Salvat:", calibration_path(video), "| verificare vizuala:", preview)
