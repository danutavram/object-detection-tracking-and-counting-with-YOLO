# speed_estimator.py
# Estimarea vitezei unui vehicul:
#  - se foloseste PUNCTUL DE CONTACT CU SOLUL (mijlocul laturii de jos a casetei), singurul punct al casetei
#    aflat in planul drumului, pentru care transformarea in metri este valabila;
#  - pozitia lui este corectata cu deplasarea camerei si transformata in metri (calibration.py);
#  - viteza rezulta din regresia liniara a pozitiei pe drum, X(t) si Y(t), pe toate cadrele valide (ambele
#    coordonate, pentru ca drumul poate fi in curba sau oblic fata de axele calibrarii);
#  - cat timp vehiculul intra sau iese din cadru caseta este taiata de marginea imaginii, iar punctul de
#    contact este fals: aceste cadre nu se folosesc;
#  - cand vehiculul e partial ascuns de vehiculul din fata (coloana deasa), caseta acopera doar partea vizibila si
#    latura ei de jos nu e pe sol; pe masura ce vehiculul iese de dupa celalalt, latura de jos coboara mai repede decat
#    vehiculul, iar viteza iese prea mare (pe video_test4: 100 km/h intr-o coloana de ~45 km/h). Aceste cadre nu se
#    folosesc: punctul de contact cade in caseta unui vehicul mai apropiat SI caseta e mai scurta decat forma
#    obisnuita a aceluiasi vehicul;
#  - viteza se raporteaza doar dupa ce estimarea a devenit stabila.
import numpy as np

BORDER_BOTTOM, BORDER_TOP, BORDER_SIDE = 0.07, 0.02, 0.005   # fasii ignorate la margini (fractiuni din imagine)
MIN_POINTS, MIN_DURATION_S = 10, 1.0     # numar minim de masuratori / durata minima de observare
MAX_RELATIVE_ERROR = 0.03                # eroarea standard a pantei, raportata la panta
SPEED_MIN, SPEED_MAX = 5, 250            # filtru de plauzibilitate (km/h)
OCCLUDER_MARGIN = 0.1     # punctul de contact e considerat ascuns si putin deasupra casetei vehiculului din fata
SHORT_BOX = 0.85          # caseta „scurta”: inaltime/latime sub 85% din percentila 90 a raportului pe aceeasi pista


def ground_point(x, y, w, h):
    """Punctul de contact cu solul al vehiculului: mijlocul laturii inferioare a casetei."""
    return x + w / 2.0, y + h


def ground_hidden(gx, gy, other_boxes):
    """Adevarat daca punctul de contact (gx, gy) cade in caseta (x, y, w, h) a unui vehicul mai apropiat de camera
    (cu latura de jos mai jos in imagine), deci poate fi ascuns de acesta."""
    return any(y + h > gy and x <= gx <= x + w and gy >= y - OCCLUDER_MARGIN * h for x, y, w, h in other_boxes)


def fully_visible(x, y, w, h, frame_w, frame_h):
    """Adevarat daca vehiculul este in intregime in cadru (caseta nu atinge fasiile de la margini)."""
    return (y >= BORDER_TOP * frame_h and y + h <= (1 - BORDER_BOTTOM) * frame_h
            and x >= BORDER_SIDE * frame_w and x + w <= (1 - BORDER_SIDE) * frame_w)


def speed_from_track(samples, aspects=None):
    """
    samples: lista de (t [s], X [m], Y [m], inaltime/latime, punct ascuns) - pozitia pe drum la fiecare cadru valid.
    aspects: raportul inaltime/latime al casetei in toate cadrele pistei (forma obisnuita a vehiculului).
    Returneaza viteza in km/h sau None daca estimarea nu este stabila.
    """
    if aspects:
        limit = SHORT_BOX * np.percentile(aspects, 90)
        samples = [s for s in samples if not (s[4] and s[3] < limit)]
    if len(samples) < MIN_POINTS:
        return None
    t, X, Y = np.array([s[:3] for s in samples]).T
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
