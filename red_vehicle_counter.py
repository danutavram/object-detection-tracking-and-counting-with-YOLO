# red_vehicle_counter.py

import cv2
import numpy as np

# Functia 'is_vehicle_red' determina daca un vehicul este de culoare rosie
def is_vehicle_red(vehicle_roi, min_red_pixel_ratio=0.13):
    # Verifica daca regiunea de interes (ROI) a vehiculului este goala
    if vehicle_roi.size == 0:
        return False  # Daca ROI-ul este gol, returneaza False

    # Convertește imaginea ROI-ului din BGR în HSV pentru a facilita detectarea culorii
    hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)

    # Definirea intervalelor de culoare pentru detectarea rosu in HSV
    lower_red1 = np.array([0, 50, 50])      # Interval pentru rosul cu nuante mici de hue (0-15)
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])    # Interval pentru rosul cu nuante mari de hue (165-180)
    upper_red2 = np.array([180, 255, 255])

    # Crearea mastii pentru zonele rosii din imagine folosind cele doua intervale
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)  # Combinarea celor doua masti pentru a acoperi tot spectrul de rosu

    # Calcularea numarului de pixeli rosii din imagine
    red_pixel_count = np.sum(red_mask > 0)
    # Calcularea numarului total de pixeli din imagine (pe baza canalului de culoare BGR)
    total_pixels = vehicle_roi.size // 3  # ROI-ul are 3 canale (B, G, R), deci se imparte la 3

    # Calcularea raportului de pixeli rosii in comparatie cu numarul total de pixeli
    red_pixel_ratio = red_pixel_count / total_pixels

    # Returneaza True daca raportul de pixeli rosii este mai mare sau egal cu pragul specificat, altfel False
    return red_pixel_ratio >= min_red_pixel_ratio