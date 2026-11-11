# Sistem de detecție, numărare și estimare a vitezei autovehiculelor

Proiect de diplomă – Avram Nicolae Dan, Universitatea din Oradea (RST), coordonator Prof. univ. dr. ing. Cristian Grava.
Limba de lucru cu utilizatorul: română.

## Reguli stabilite de utilizator

1. **O singură versiune a codului.** Fără fișiere `_v2`, fără variante paralele, fără cod vechi păstrat „pentru comparație”.
2. **Lucrarea scrisă (docx) nu se modifică** până când utilizatorul nu confirmă că aceasta este versiunea finală a codului.
   Lucrarea se generează în altă conversație (Claude, aplicația desktop), nu din acest folder.
3. Programul trebuie să funcționeze **pe orice videoclip cu cameră fixă**, nu doar pe `video0.mp4`, cu viteza cât mai exactă.
4. Urmează (necerut încă în cod): citirea numerelor de înmatriculare, pe clipuri filmate suficient de aproape.

## Structura

| Fișier | Rol |
|---|---|
| `main.py` | rulare fără interfață pe `video0.mp4` |
| `gui.py` | interfața CustomTkinter + fereastra de calibrare (`CalibrationWindow`) |
| `main_tracker.py` | bucla principală: cadru → detecție+urmărire → stabilizare → numărare/viteză/roșu → adnotare → video + CSV |
| `model_loader.py` | încarcă modelul (`MODEL_PATH = "yolo26s.pt"`) |
| `vehicle_detector.py` | `model.track(..., tracker="bytetrack.yaml", classes=[2,3,5,7], conf=0.5)` |
| `tracker.py` | clasa `Tracker`: numărare la linia din mijlocul imaginii, sens din direcția de mișcare, roșu, viteze |
| `auto_calibration.py` | calibrare automată din marcajele discontinue (fundal → punct de fugă → raze → liniuțe periodice → model de cameră) |
| `calibration.py` | clasa `Calibration`, calibrare automată/manuală, salvare `<video>.calib.json`, `Stabilizer`, grila de verificare, CLI |
| `speed_estimator.py` | punct de contact cu solul → metri → regresie liniară Y(t); afișare doar când estimarea e stabilă |
| `red_vehicle_counter.py` | prag HSV pentru roșu (proporție ≥ 0,13) |
| `video_manager.py` | deschidere video / VideoWriter |
| `test_sistem.py`, `ground_truth.json` | test automat pe `video0.mp4`: calibrare automată + 68 de treceri adnotate manual |

Contractul cu interfața: `real_time_detection_with_callback(video_path, output_path, frame_callback=None, stop_event=None)`;
`frame_callback(frame, stats)` cu cheile `entry_count` (se apropie), `exit_count` (se depărtează), `red_vehicle_count`,
`max_speed`, `frame_number`, `total_frames`, `calibrated`.

## Cum funcționează viteza

- Fără fișier `<video>.calib.json` programul numără și detectează, dar **nu afișează viteze**.
- **Calibrare automată** (`python calibration.py clip.mp4 --marcaj PERIOADA LINIUTA BANDA` sau din interfață): găsește liniile
  discontinue; dimensiunile reale ale marcajului trebuie date de utilizator – geometria nu le poate deduce (verificat: orice
  standard se potrivește la fel de bine, distanța focală absoarbe diferența). Dimensiuni greșite = toate vitezele greșite cu același procent.
- **Calibrare manuală** (interfață): 4 click-uri pe un dreptunghi de pe drum + lățime și lungime în metri.
- **Stabilizare**: deplasarea camerei față de fundal se măsoară la fiecare 5 cadre (Lucas-Kanade) și se scade din poziția vehiculului.
  În `video0.mp4` camera alunecă ≈21 px în primele 3 secunde; fără stabilizare două vehicule aveau erori de până la 15 %.
- **Intrarea în cadru**: cât timp caseta atinge fâșiile de la margine (7 % jos, 2 % sus) pozițiile nu se folosesc; viteza se
  afișează după ≥1 s, ≥10 puncte și eroare standard a pantei ≤3 %. Înainte, prima valoare afișată era în medie cu 12 % prea mică.
- Punctul folosit este mijlocul laturii de jos a casetei (contactul cu solul), nu centrul casetei.

## Rezultate (rulate pe Linux, CPU; numărarea de pe clipurile de test e verificată doar vizual, parțial)

Pe Windows a fost re-rulat doar `test_sistem.py` (`video0.mp4`): rezultate identice. Clipurile de test nu au fost re-rulate pe Windows.

| Clip | Marcaj folosit | Calibrare | Numărare | Viteze |
|---|---|---|---|---|
| `video0.mp4` | 18 / 6 / 3,75 (autostradă DE) | automată, 4 linii, eroare 0,45 m | 34 + 34, exact față de referință; 5 roșii din 5 | 68 din 68; 75–187 km/h; față de calibrarea manuală veche: diferență mediană 0,5 %, max. 1,9 % (după primele 3 s) |
| `video_test1.mp4` | 9 / 2 / 3,65 (autostradă UK) | automată, 4 linii, 0,28 m | 56 + 49 | 104 din 105; 31–113 km/h |
| `video_test2.mp4` | 15 / 6 / 3,75 (**presupus**, standard necunoscut) | automată, 2 linii, 0,19 m | 15 (un vehicul ratat) | 15 din 15; 59–96 km/h |
| `video_test3.mp4` | – | automată nereușită (curbă, o singură linie discontinuă) | 45 | necesită calibrare manuală |

## Probleme cunoscute / de făcut

1. Interfața a fost rulată pe Windows (11.11.2026) cu acțiuni simulate din cod: calibrare automată, manuală cu 4 click-uri,
   salvare, START/STOP, fișier inexistent/necitibil – toate merg. Calculele lungi din fereastra de calibrare rulează într-un
   thread (`_run_async`); thread-ul de procesare nu atinge interfața, sfârșitul e tratat în `_poll`/`_finish`.
   Rămas: utilizatorul nu a calibrat încă manual `video_test3` cu un dreptunghi real (click-urile din test au fost arbitrare).
   Pregătirea la START durează ≈15 s (fundalul pentru stabilizare se recalculează de fiecare dată: 60 de salturi în clip).
2. `video_test2`: o autoutilitară văzută de sus e clasificată „train” la linia de numărare și nu e numărată. De încercat:
   prag de încredere mai mic (0,25) cu ByteTrack, cu re-verificare pe `video0.mp4`.
3. `video_test1`: 13 vehicule roșii pare prea mult (lumină de apus); de verificat pragul HSV.
4. `auto_calibration.py` este scris compact (prototip validat); de făcut mai lizibil fără a schimba comportamentul
   (testul `test_sistem.py` trebuie să treacă în continuare).
5. Presetări de marcaj pentru România (de luat din standardul SR 1848-7, nu din memorie).
6. Numere de înmatriculare: pe `video_test3` plăcuțele au ≈45 px lățime – la limită.
7. `all_code.py` este o concatenare veche a codului; de șters dacă utilizatorul e de acord.
8. Captură de ecran a interfeței pentru lucrare. Timpul pe cadru pe PC-ul utilizatorului (Windows, CPU, fără CUDA):
   ≈101–105 ms/cadru pe `video0.mp4` (1080p), măsurat cu `test_sistem.py`.

## Mediul Python (Windows)

`python -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt` → `python test_sistem.py` → `python gui.py`.
