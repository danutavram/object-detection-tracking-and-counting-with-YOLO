# test_sistem.py
# Test complet al sistemului pe video0.mp4: calibrare automata + tot lantul de prelucrare (fara interfata grafica)
# si compara rezultatele cu adevarul de referinta (ground_truth.json - 68 de treceri adnotate manual).
# Rulare:  python test_sistem.py
import json
import sys
import time

from calibration import calibrate_auto, save_calibration
from main_tracker import real_time_detection_with_callback

VIDEO, OUTPUT = "video0.mp4", "output_test.mp4"
TOLERANCE = 20          # cadre de toleranta in jurul intervalului adnotat
EXPECTED_RED = 5
MARKING = (18.0, 6.0, 3.75)   # autostrada germana: perioada 18 m, liniuta 6 m, banda 3,75 m


def match(crossings, reference):
    """Asociaza trecerile numarate cu cele de referinta (una la una, in ordine cronologica)."""
    free = sorted(crossings)
    missed = []
    for start, end, lane, note in reference:
        hit = next((f for f in free if start - TOLERANCE <= f <= end + TOLERANCE), None)
        if hit is None:
            missed.append((start, end, lane, note))
        else:
            free.remove(hit)
    return len(reference) - len(missed), free, missed


def main():
    gt = json.load(open("ground_truth.json", encoding="utf-8"))
    calib = calibrate_auto(VIDEO, *MARKING)
    save_calibration(VIDEO, calib)
    print(f"{calib.describe()}: {calib.info['lines']} linii discontinue, {calib.info['points']} puncte, "
          f"camera la {calib.info['camera_height_m']:.1f} m")
    ok_calib = calib.info['lines'] == 4 and calib.info['rmse'] < 0.6

    t0 = time.time()
    r = real_time_detection_with_callback(VIDEO, OUTPUT)
    if r is None:
        sys.exit("EROARE: videoclipul nu a putut fi deschis.")
    dt = time.time() - t0

    ok = ok_calib
    for name, direction, key in (("stang (se apropie)", "in", "L"), ("drept (se departeaza)", "out", "R")):
        frames = [r["cross_frame"][i] for i, d in r["counted"].items() if d == direction]
        tp, extra, missed = match(frames, gt[key])
        print(f"Sens {name}: numarate {len(frames)}, referinta {len(gt[key])}, "
              f"corecte {tp}, false {len(extra)}, ratate {len(missed)}")
        for m in missed:
            print("   ratat:", m)
        for f in extra:
            print("   fals la cadrul:", f)
        ok &= not extra and not missed

    print(f"Vehicule rosii: {r['red_vehicle_count']} (referinta {EXPECTED_RED})")
    ok &= r["red_vehicle_count"] == EXPECTED_RED

    counted_with_speed = [r["speeds"][i] for i in r["counted"] if i in r["speeds"]]
    print(f"Viteza masurata pentru {len(counted_with_speed)} din {len(r['counted'])} vehicule numarate; "
          f"min {min(counted_with_speed):.0f}, max {max(counted_with_speed):.0f} km/h")
    ok &= len(counted_with_speed) == len(r["counted"])

    print(f"Cadre procesate: {r['frames']} in {dt:.0f} s ({1000 * dt / r['frames']:.0f} ms/cadru)")
    print("REZULTAT:", "TOATE VERIFICARILE AU TRECUT" if ok else "EXISTA DIFERENTE FATA DE REFERINTA")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
