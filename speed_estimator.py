# speed_estimator.py
# Estimarea vitezei unui vehicul:
#  - se foloseste PUNCTUL DE CONTACT CU SOLUL (mijlocul laturii de jos a casetei), singurul punct al casetei
#    aflat in planul drumului, pentru care transformarea in metri este valabila;
#  - pozitia lui este corectata cu deplasarea camerei si transformata in metri (calibration.py);
#  - viteza este panta dreptei Y(t), obtinuta prin regresie liniara pe toate cadrele valide;
#  - cat timp vehiculul intra sau iese din cadru caseta este taiata de marginea imaginii, iar punctul de
#    contact este fals: aceste cadre nu se folosesc;
#  - viteza se raporteaza doar dupa ce estimarea a devenit stabila.
import numpy as np

BORDER_BOTTOM, BORDER_TOP, BORDER_SIDE = 0.07, 0.02, 0.005   # fasii ignorate la margini (fractiuni din imagine)
MIN_POINTS, MIN_DURATION_S = 10, 1.0     # numar minim de masuratori / durata minima de observare
MAX_RELATIVE_ERROR = 0.03                # eroarea standard a pantei, raportata la panta
SPEED_MIN, SPEED_MAX = 5, 250            # filtru de plauzibilitate (km/h)


def ground_point(x, y, w, h):
    """Punctul de contact cu solul al vehiculului: mijlocul laturii inferioare a casetei."""
    return x + w / 2.0, y + h


def fully_visible(x, y, w, h, frame_w, frame_h):
    """Adevarat daca vehiculul este in intregime in cadru (caseta nu atinge fasiile de la margini)."""
    return (y >= BORDER_TOP * frame_h and y + h <= (1 - BORDER_BOTTOM) * frame_h
            and x >= BORDER_SIDE * frame_w and x + w <= (1 - BORDER_SIDE) * frame_w)


def speed_from_track(samples):
    """
    samples: lista de (t [s], Y [m]) - pozitia in lungul drumului la fiecare cadru valid.
    Returneaza viteza in km/h sau None daca estimarea nu este inca stabila.
    """
    if len(samples) < MIN_POINTS:
        return None
    t = np.array([s[0] for s in samples])
    Y = np.array([s[1] for s in samples])
    if t.max() - t.min() < MIN_DURATION_S:
        return None
    slope, intercept = np.polyfit(t, Y, 1)
    residuals = Y - (slope * t + intercept)
    slope_error = np.sqrt(np.sum(residuals ** 2) / (len(t) - 2) / np.sum((t - t.mean()) ** 2))
    if abs(slope) < 1e-6 or slope_error / abs(slope) > MAX_RELATIVE_ERROR:
        return None
    v = abs(slope) * 3.6
    return float(v) if SPEED_MIN < v < SPEED_MAX else None
