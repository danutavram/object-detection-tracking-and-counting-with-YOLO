# Sistem de detecție, numărare și estimare a vitezei autovehiculelor

Proiect de diplomă – Avram Nicolae Dan, Universitatea din Oradea (RST), coordonator Prof. univ. dr. ing. Cristian Grava.
Limba de lucru cu utilizatorul: română.

## Reguli stabilite de utilizator

1. **O singură versiune a codului.** Fără fișiere `_v2`, fără variante paralele, fără cod vechi păstrat „pentru comparație”.
2. **Lucrarea scrisă (docx) nu se modifică** până când utilizatorul nu confirmă că aceasta este versiunea finală a codului.
   Lucrarea se generează în altă conversație (Claude, aplicația desktop), nu din acest folder.
3. Programul trebuie să funcționeze **pe orice videoclip cu cameră fixă**, nu doar pe `video0.mp4`, cu viteza cât mai exactă.
   Utilizatorul **nu știe din ce țară e drumul**: programul nu are voie să-i ceară tipul marcajului; trebuie să-l afle singur.
4. Citirea numerelor de înmatriculare (LPR) pe clipuri filmate suficient de aproape: implementată (`plate_reader.py`).

5. **Lucrarea și proiectul respectă procedura UO SEAQ_PO_Pr.MA_06** (elaborarea lucrării de finalizare a studiilor).
   Cerințele, componentele preluate cu licențele lor, sursele videoclipurilor și afirmațiile încă nesusținute sunt în
   `CONFORMITATE_PROCEDURA_UO.md`; fișierul se actualizează la fiecare componentă, clip sau constantă nouă.
   Consecințe pentru cod: nicio valoare „din memorie” prezentată ca fapt (se notează sursa sau că e neverificată);
   niciun rezultat raportat fără a spune cum a fost verificat; orice bibliotecă nouă se trece cu licența ei.
   Deschis: dacă asistența AI e permisă o hotărăște coordonatorul (declarația de autenticitate, Anexa 10).

## Structura

| Fișier | Rol |
|---|---|
| `main.py` | rulare fără interfață pe `video0.mp4` |
| `gui.py` | interfața CustomTkinter + fereastra de calibrare (`CalibrationWindow`) |
| `main_tracker.py` | bucla principală: cadru → detecție+urmărire → stabilizare → numărare/viteză/roșu → adnotare → video + CSV |
| `model_loader.py` | încarcă modelul (`MODEL_PATH = "yolo26s.pt"`) |
| `vehicle_detector.py` | `model.track(..., tracker="bytetrack.yaml", classes=[2,3,5,7], conf=0.5)` |
| `tracker.py` | clasa `Tracker`: numărare la linia din mijlocul imaginii, sens din direcția de mișcare, roșu, viteze |
| `auto_calibration.py` | geometria din marcaje: mai multe linii discontinue pe drum drept (fundal → punct de fugă → raze → liniuțe periodice → model de cameră) sau o singură linie, eventual în curbă (lanț de liniuțe); estimarea scării din casetele mașinilor |
| `calibration.py` | clasa `Calibration`, tabelul `MARKING_STANDARDS`, identificarea automată a marcajului, calibrare automată/manuală, salvare `<video>.calib.json`, `Stabilizer`, grila de verificare, CLI |
| `speed_estimator.py` | punct de contact cu solul → metri → regresie liniară X(t), Y(t) (2D, pentru drum în curbă); afișare doar când estimarea e stabilă |
| `plate_reader.py` | LPR cu `fast-alpr` (detector de plăcuțe + OCR, ONNX): fiecare vehicul e citit în mai multe cadre, numărul final prin vot pe caractere; apare în etichetă și în CSV |
| `red_vehicle_counter.py` | prag HSV pentru roșu (proporție ≥ 0,13) |
| `video_manager.py` | deschidere video / VideoWriter |
| `test_sistem.py`, `ground_truth.json` | test automat pe `video0.mp4`: calibrare automată + 68 de treceri adnotate manual |

Contractul cu interfața: `real_time_detection_with_callback(video_path, output_path, frame_callback=None, stop_event=None)`;
`frame_callback(frame, stats)` cu cheile `entry_count` (se apropie), `exit_count` (se depărtează), `red_vehicle_count`,
`max_speed`, `frame_number`, `total_frames`, `calibrated`.

## Cum funcționează viteza

- Fără fișier `<video>.calib.json` programul numără și detectează, dar **nu afișează viteze**.
- **Calibrare automată** (`python calibration.py clip.mp4` sau din interfață, opțiunea implicită „Automat”):
  1. geometria din marcaje. Cu ≥2 linii discontinue: model de cameră complet. Cu o singură linie (`video_test3`): lanț de
     liniuțe egale și echidistante, cameră fără rotire în jurul axei optice; focala nu e determinată de o singură linie.
  2. tipul marcajului (dimensiunile reale) e ales din `MARKING_STANDARDS` după două măsurători: raportul liniuță/perioadă
     (măsurat direct, precis; corecție de +0,035 / +0,015 pentru neclaritate) și perioada estimată din mașini (lățimea
     aparentă a casetei + cât din flanc se vede pe benzile laterale). `info["confidence"]` = „ridicata” / „scazuta”.
  Limită fizică, verificată: la filmări cu teleobiectiv, marcajele singure nu dau scara în lungul drumului (focala și perioada
  se compensează exact), iar înălțimea/lungimea casetelor nu discriminează între standarde. Estimarea din flanc are eroare de
  ≈5–30 % (9,4 m față de 9 pe `video_test1`; 23,2 față de 18 pe `video0`), deci singură nu ajunge: de aceea se combină cu
  raportul liniuță/perioadă și se alege un standard din tabel. Valorile din tabel sunt din memorie, neverificate în normative.
  Utilizatorul poate impune un tip din listă; dacă raportul lui nu corespunde cu imaginea, interfața avertizează.
- **Calibrare manuală** (interfață): 4 click-uri pe un dreptunghi de pe drum + lățime și lungime în metri.
- **Stabilizare**: deplasarea camerei față de fundal se măsoară la fiecare 5 cadre (Lucas-Kanade) și se scade din poziția vehiculului.
  În `video0.mp4` camera alunecă ≈21 px în primele 3 secunde; fără stabilizare două vehicule aveau erori de până la 15 %.
- **Intrarea în cadru**: cât timp caseta atinge fâșiile de la margine (7 % jos, 2 % sus) pozițiile nu se folosesc; viteza se
  afișează după ≥1 s, ≥10 puncte și eroare standard a pantei ≤3 %. Înainte, prima valoare afișată era în medie cu 12 % prea mică.
- Punctul folosit este mijlocul laturii de jos a casetei (contactul cu solul), nu centrul casetei.

## Rezultate (Windows, CPU, 14.11.2026; numărarea de pe clipurile de test e verificată doar vizual, parțial)

| Clip | Marcaj identificat automat | Calibrare | Numărare | Viteze |
|---|---|---|---|---|
| `video0.mp4` | Germania, autostradă 6+12 m (corect), încredere ridicată | 4 linii, eroare 0,45 m | 34 + 34, exact față de referință; 5 roșii din 5 | 68 din 68; 75–187 km/h |
| `video_test1.mp4` | Marea Britanie 2+7 m (corect), încredere ridicată | 4 linii, 0,28 m | 56 + 49 | 104 din 105; 31–113 km/h; camioane cel mult 98 |
| `video_test2.mp4` | „6 m + 9 m”, **încredere scăzută** (standard real necunoscut; din mașini 13,9 m) | 2 linii, 0,19 m | 15 (un vehicul ratat) | 15 din 15; 59–95 km/h |
| `video_test3.mp4` | „Polonia urban 2+4 m”, **încredere scăzută** (din mașini 6,6 m). Instabil: la mici schimbări ale măsurării raportului a ieșit pe rând 7,5 / 9 / 6 m | 1 linie (lanț de 5 liniuțe), 0,43 m | 45 | nerulat cu ultima calibrare |
| `video_test4.mp4` | Marea Britanie 2+7 m, **încredere scăzută** (ales doar din raportul liniuță/perioadă 0,21; mașinile nu determină scara la teleobiectiv). Coerent cu plăcuțele britanice | 1 linie (24 de liniuțe scurte, `straight_dash_line`), 0,82 m; cameră la ≈6 m | 12 la cadrul 642 (rulat parțial din interfață) | 39–63 km/h în trafic dens |

Tirul cu 180 km/h de pe `video_test1` venea din calibrarea făcută în interfață cu marcajul implicit german (perioadă 18 m în loc de 9 m).

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
5. Tabelul `MARKING_STANDARDS` e din memorie: de verificat în normative și de adăugat România (SR 1848-7).
   Pe clipurile cu încredere scăzută (`video_test2`, `video_test3`) scara poate fi greșită cu zeci de procente;
   pentru precizie rămâne calibrarea manuală. De căutat o referință longitudinală mai bună (ex. ampatamentul).
6. LPR în interfață: celulă „Numere citite” + listă sub statistici (`plates_box`); în imagine numărul e scris mare sub casetă.
   Cheia `plates` din `stats` = listă de (id, număr, viteză). Citirile timpurii pot fi greșite până se adună voturi (ex. „417WW” → „EY09VWS”).
   Pe `video_test4.mp4` (plăcuțe britanice, 80–100 px): 19 numere citite din 26 de vehicule numărate; verificate vizual
   doar câteva (LN15ZZC, KH06KSU, GJ06EPD, EY09VWS, DA07CLX – corecte). Două citiri sunt trunchiate (5 caractere din 7).
   Nu există referință adnotată, deci acuratețea nu e măsurată. Cu LPR procesarea e ≈216 ms/cadru pe acest clip.
   Atenție: `fast-alpr` instalează `opencv-python-headless`, care suprascrie `opencv-python`; după `pip install -r
   requirements.txt` se rulează `pip uninstall -y opencv-python-headless` și `pip install --force-reinstall --no-deps opencv-python`.
7. `all_code.py` este o concatenare veche a codului; de șters dacă utilizatorul e de acord.
8. Captură de ecran a interfeței pentru lucrare. Timpul pe cadru pe PC-ul utilizatorului (Windows, CPU, fără CUDA):
   ≈101–105 ms/cadru pe `video0.mp4` (1080p), măsurat cu `test_sistem.py`.

## Mediul Python (Windows)

`python -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt` → `python test_sistem.py` → `python gui.py`.
