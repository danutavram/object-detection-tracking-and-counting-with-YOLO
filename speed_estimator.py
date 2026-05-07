# speed_estimator.py

import cv2
import numpy as np

# Scala bird's-eye view: 10 pixeli = 1 metru
BEV_SCALE = 10.0

# Latimea reala a celor 3 benzi (3 x 3.75m pe autobahn)
LANE_WIDTH_METERS = 11.25

# Puncte de referinta pe marginile drumului (identificate din marcajele albe)
# Format: (x, y) in pixeli din cadrul video
# Sens STANGA (vehicule care se departeaza de camera)
SRC_LEFT = np.float32([
    [306, 500],   # marginea stanga la y=500
    [806, 500],   # marginea dreapta la y=500
    [53, 800],    # marginea stanga la y=800
    [692, 800],   # marginea dreapta la y=800
])

# Sens DREAPTA (vehicule care se apropie de camera)
SRC_RIGHT = np.float32([
    [1037, 500],  # marginea stanga la y=500
    [1647, 500],  # marginea dreapta la y=500
    [1138, 800],  # marginea stanga la y=800
    [1672, 800],  # marginea dreapta la y=800
])

# Destinatie: dreptunghi in spatiul real (bird's-eye view)
DST_RECT = np.float32([
    [0, 0],
    [LANE_WIDTH_METERS * BEV_SCALE, 0],
    [0, 100 * BEV_SCALE],
    [LANE_WIDTH_METERS * BEV_SCALE, 100 * BEV_SCALE],
])

# Separator intre sensuri (pixeli x)
MIDLINE_X = 915


def compute_homographies():
    """Calculeaza matricile de homografie pentru cele doua sensuri."""
    H_left, _ = cv2.findHomography(SRC_LEFT, DST_RECT)
    H_right, _ = cv2.findHomography(SRC_RIGHT, DST_RECT)
    return H_left, H_right


def transform_point(x, y, H):
    """Transforma un punct din spatiul video in spatiul real (metri)."""
    pt = np.float32([[[x, y]]])
    transformed = cv2.perspectiveTransform(pt, H)[0][0]
    return transformed[0] / BEV_SCALE, transformed[1] / BEV_SCALE


def get_real_world_position(cx, cy, H_left, H_right):
    """Returneaza pozitia reala (metri) si sensul de mers."""
    if cx < MIDLINE_X:
        rx, ry = transform_point(cx, cy, H_left)
        return rx, ry, 'left'
    else:
        rx, ry = transform_point(cx, cy, H_right)
        return rx, ry, 'right'


def calculate_real_distance(pos1, pos2):
    """Calculeaza distanta reala in metri intre doua pozitii transformate."""
    return np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)
