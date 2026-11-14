# speed_estimator.py
# Estimarea vitezei unui vehicul:
#  - se foloseste PUNCTUL DE CONTACT CU SOLUL (mijlocul laturii de jos a casetei), singurul punct al casetei
#    aflat in planul drumului, pentru care transformarea in metri este valabila;
#  - pozitia lui este corectata cu deplasarea camerei si transformata in metri (calibration.py);
#  - viteza rezulta din regresia liniara a pozitiei pe drum, X(t) si Y(t), pe toate cadrele valide (ambele
#    coordonate, pentru ca drumul poate fi in curba sau oblic fata de axele calibrarii);
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
    samples: lista de (t [s], X [m], Y [m]) - pozitia pe drum la fiecare cadru valid.
    Returneaza viteza in km/h sau None daca estimarea nu este inca stabila.
    """
    if len(samples) < MIN_POINTS:
        return None
    t, X, Y = np.array(samples).T
    if t.max() - t.min() < MIN_DURATION_S:
        return None
    (vx, x0), (vy, y0) = np.polyfit(t, X, 1), np.polyfit(t, Y, 1)
    speed = np.hypot(vx, vy)                                   # m/s
    # eroarea standard a vitezei: abaterile de la miscarea uniforma, masurate pe directia de mers
    along = ((X - (vx * t + x0)) * vx + (Y - (vy * t + y0)) * vy) / max(speed, 1e-9)
    speed_error = np.sqrt(np.sum(along ** 2) / (len(t) - 2) / np.sum((t - t.mean()) ** 2))
    if speed < 1e-6 or speed_error / speed > MAX_RELATIVE_ERROR:
        return None
    v = speed * 3.6
    return float(v) if SPEED_MIN < v < SPEED_MAX else None
