# imagine_timp.py
# Imagine-timp („slit-scan”) pentru numararea manuala a vehiculelor: din fiecare cadru se ia acelasi rand de pixeli si
# randurile se pun unul sub altul. Fiecare vehicul care trece randul apare ca o pata separata; timpul creste in jos.
# Pentru o coloana de masini care se acopera, randul se alege jos in cadru, unde masinile sunt despartite de asfalt.
# Imaginea se imparte in bucati (PNG) cu marcaje la fiecare secunda; numaratoarea se face uitandu-te la ele.
#
# Utilizare (din folderul proiectului):
#   python experimente/imagine_timp.py video_test4 0.88 250 1650 experimente/imagini_timp/video_test4
#   argumente: videoclip (fara .mp4), randul (fractiune din inaltime), x de inceput si de sfarsit (pixeli),
#   prefixul fisierelor rezultate. Optional: --bucata N (cadre pe imagine, implicit 430).
import os
import sys

import cv2
import numpy as np

args = sys.argv[1:]
chunk = 430
if "--bucata" in args:
    i = args.index("--bucata")
    chunk = int(args[i + 1])
    del args[i:i + 2]
video, y_frac, x0, x1, out = args[0], float(args[1]), int(args[2]), int(args[3]), args[4]
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

cap = cv2.VideoCapture(video + ".mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
rows = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    rows.append(frame[int(y_frac * frame.shape[0]), x0:x1])
img = np.array(rows)
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0]))          # latimea /2, un rand pe cadru

for start in range(0, len(img), chunk):
    part = cv2.resize(img[start:start + chunk], None, fx=1, fy=2, interpolation=cv2.INTER_NEAREST)
    ticks = np.zeros((part.shape[0], 70, 3), np.uint8)
    for f in range(start, min(start + chunk, len(img))):
        if f % round(fps) == 0:
            y = (f - start) * 2
            cv2.line(ticks, (50, y), (70, y), (255, 255, 255), 1)
            cv2.putText(ticks, f"{f / fps:.0f}s", (2, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    name = f"{out}_{start // chunk}.png"
    cv2.imwrite(name, np.hstack([ticks, part]))
    print(name)
