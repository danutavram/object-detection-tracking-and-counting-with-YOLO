# red_vehicle_counter.py
# Decide daca un vehicul este rosu, dupa culoarea pixelilor din caseta lui, in spatiul de culoare CIE Lab.
#
# - Culoarea se masoara relativ la dominanta de culoare a scenei (mediana a*, b* a cadrului): o camera cu balansul de
#   alb deplasat (ex. video_test1, la apus, cu tenta albastra) muta toate culorile in aceeasi directie.
# - Un pixel este „rosu” daca are croma relativa >= MIN_CHROMA si unghiul nuantei (atan2(b*, a*)) intre
#   HUE_MIN si HUE_MAX grade. Limita de sus exclude portocaliul (ex. autocamionul TNT de pe video_test1), limita de jos
#   magenta. Pixelii foarte intunecati (L* < MIN_L) nu au o culoare sigura si nu se folosesc.
# - Vehiculul este rosu daca proportia pixelilor rosi din partea vizibila a casetei este >= MIN_RED_RATIO. Partea
#   acoperita de casetele vehiculelor mai apropiate de camera nu se foloseste (altfel, pe video_test4, masina rosie
#   ascunsa pe jumatate de una alba iesea 0,09 in loc de 0,19); daca se vede mai putin de MIN_VISIBLE din caseta,
#   se foloseste toata caseta.
# Pragurile sunt alese pe vehiculele numarate din video0, video_test3 si video_test1, clasificate manual (5, 5 si 7
# vehicule rosii): rosiile au proportia >= 0,22, celelalte <= 0,12 (cu stopuri aprinse incluse). Nu sunt validate pe
# un set independent (CLAUDE.md, „Rosu”).
import cv2
import numpy as np

MIN_CHROMA = 10          # croma relativa minima (unitati Lab)
HUE_MIN, HUE_MAX = -20, 30   # unghiul nuantei rosii (grade)
MIN_L = 20               # luminozitate minima (L* pe scara 0-255 din OpenCV)
MIN_RED_RATIO = 0.17     # proportia minima de pixeli rosii din partea vizibila a casetei
MIN_VISIBLE = 0.3        # sub aceasta parte vizibila se foloseste toata caseta


def scene_color_cast(frame):
    """Dominanta de culoare a scenei: mediana lui a* si b* pe cadrul micsorat (de 8 ori)."""
    small = cv2.resize(frame, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return float(np.median(lab[:, 1])), float(np.median(lab[:, 2]))


def is_vehicle_red(frame, box, cast=(128.0, 128.0), closer_boxes=()):
    """
    Adevarat daca vehiculul din caseta box = (x, y, w, h) a cadrului BGR este rosu.
    cast = scene_color_cast(frame); closer_boxes = casetele vehiculelor mai apropiate de camera (pot acoperi vehiculul).
    """
    H, W = frame.shape[:2]
    x, y, w, h = box
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return False
    visible = np.ones((y1 - y0, x1 - x0), bool)
    for bx, by, bw, bh in closer_boxes:
        visible[max(by - y0, 0):max(by + bh - y0, 0), max(bx - x0, 0):max(bx + bw - x0, 0)] = False
    if visible.mean() < MIN_VISIBLE:
        visible[:] = True
    lab = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)[visible].astype(np.float32)
    da, db = lab[:, 1] - cast[0], lab[:, 2] - cast[1]
    hue = np.degrees(np.arctan2(db, da))
    red = (np.hypot(da, db) >= MIN_CHROMA) & (hue >= HUE_MIN) & (hue <= HUE_MAX) & (lab[:, 0] >= MIN_L)
    return red.mean() >= MIN_RED_RATIO
