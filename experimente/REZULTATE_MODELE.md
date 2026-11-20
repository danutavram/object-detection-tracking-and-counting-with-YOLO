# Compararea variantelor YOLO și a pragului de încredere

**Document intern, nu intră în lucrare** (regula 6 din `CLAUDE.md`: în lucrare celelalte modele doar se menționează).
Servește la alegerea modelului și a pragului de încredere (`CLAUDE.md`, „Probleme cunoscute” 3 și 4).

Calculatorul utilizatorului: AMD Ryzen 5 5600X, 32 GB RAM, NVIDIA GeForce RTX 3060 Ti (driver 621.24), Windows 11,
Python 3.13.13, PyTorch 2.15.0+cu132, Ultralytics 8.4.210, OpenCV 5.0.0. Măsurat la 20.11.2026.

Datele brute sunt în `rezultate_modele.json`, `timp_detectie.json` și `calibrare_modele.json`, iar scripturile care le
reproduc sunt `compara_modele.py` și `timp_detectie.py`.

## 1. Timpul de detecție pe placa video (doar rețeaua, 20 de cadre din `video0.mp4`)

O trecere de încălzire, apoi mediana a 5 treceri (`ms_per_pass` în JSON); diferența dintre treceri e de cel mult ≈15 %.
Precizia pe COCO este cea publicată de Ultralytics, nu măsurată aici:
https://docs.ultralytics.com/models/yolo26/ și https://docs.ultralytics.com/models/yolo11/

| Model | ms/cadru | Vehicule detectate/cadru | Parametri (M, fișierul .pt) | mAP50-95 COCO (oficial) |
|---|---|---|---|---|
| YOLO11n | 13,5 | 7,6 | 2,6 | 39,5 |
| YOLO11s | 13,9 | 9,0 | 9,5 | 47,0 |
| YOLO11m | 17,0 | 10,8 | 20,1 | 51,5 |
| YOLO11l | 28,7 | 11,2 | 25,4 | 53,4 |
| YOLO11x | 27,3 | 13,0 | 57,0 | 54,7 |
| YOLO26n | 16,8 | 6,7 | 2,6 | 40,9 |
| YOLO26s | 15,2 | 9,8 | 10,0 | 48,6 |
| YOLO26m | 16,4 | 11,9 | 21,9 | 53,1 |
| YOLO26l | 23,0 | 10,1 | 26,3 | 55,0 |
| YOLO26x | 22,4 | 13,4 | 59,0 | 57,5 |

Nu există variantă „xl”: mărimile sunt n, s, m, l, x.
Timpul crește puțin cu mărimea modelului: pe un singur cadru placa nu e folosită din plin, iar pregătirea imaginii și
filtrarea casetelor se fac pe procesor (explicație probabilă, nemăsurată pe componente). Variantele l au ieșit cu 3–5 %
mai lente decât variantele x, în limita variației; diferența nu e explicată.

## 2. Tot programul (fără citirea numerelor), prag de încredere 0,5, ByteTrack implicit

Numărările nu depind de dispozitiv: YOLO26n și YOLO26s pe `video0`, rulate din nou pe placa video, dau exact aceleași
numere ca în tabel.

`video0.mp4`: referință de 34 + 34 de treceri (`ground_truth.json`) și 5 vehicule roșii.

| Model | Sens stâng corecte / false | Sens drept corecte / false | Roșii | Vehicule cu viteză |
|---|---|---|---|---|
| YOLO26n | 34/34 / 0 | **33/34** / 0 | 5 | 67/67 |
| YOLO26s | 34/34 / 0 | 34/34 / 0 | 5 | 68/68 |
| YOLO26m | 34/34 / 0 | 34/34 / 0 | 5 | 68/68 |
| YOLO26l | 34/34 / 0 | 34/34 / 0 | 5 | 68/68 |
| YOLO11s | 34/34 / 0 | 34/34 / 0 | 5 | 68/68 |

Vitezele aceluiași vehicul diferă între YOLO26n/s/m/l cu mai puțin de 1% pentru 90% dintre vehicule (mediana ≈ 0%).

`video_test2.mp4`: 16 vehicule, toate în același sens. Al 16-lea este autoutilitara de la 25,07 s, clasificată
„train” de YOLO26s și deci eliminată. Referința provine din compararea rulărilor și dintr-o verificare vizuală
anterioară, nu dintr-o adnotare cadru cu cadru.

| Model | Numărate | Autoutilitara de la 25,07 s |
|---|---|---|
| YOLO26s | 15 | ratată |
| YOLO26m | 16 | numărată |
| YOLO11s | 16 | numărată (63 km/h) |

Vitezele celorlalte 15 vehicule diferă între YOLO11s și YOLO26s cu cel mult 0,9%.

`video_test1.mp4` (fără referință): YOLO26s numără 56 + 49, iar YOLO11s 54 + 49. Cele două treceri în plus ale
YOLO26s (sensul „se apropie”, 20,34 s și 35,29 s) au fost verificate în cadre: sunt o mașină roșie și o autoutilitară
albastră reale, pe care YOLO11s nu le detectează când trec linia. Nu sunt numărări duble.

`video_test3.mp4` și `video_test4.mp4`, linia fixă 0,50: YOLO11s 41 / 31, YOLO26s 45 / 26, YOLO11s + `bytetrack_strict`
40 / 30, YOLO26s + `bytetrack_strict` 43 / 20. Pe `video_test4` toate numără mult sub cele ≈48 de vehicule reale;
cauza e poziția liniei, nu modelul (vezi secțiunea 5).

## 5. Înălțimea liniei de numărare (YOLO26s / YOLO11s, aceeași rulare, 13 linii)

| Clip | 0,50 | 0,55 | 0,60 | 0,65 | 0,70 | 0,75 | 0,80 |
|---|---|---|---|---|---|---|---|
| `video0` (referință 34 + 34) | 34+34 / 34+34 | 34+34 / 34+34 | 31+34 / 30+34 | 29+34 / 29+34 | 27+32 / 27+32 | 27+31 / 27+29 | 27+25 / 27+26 |
| `video_test1` | 105 / 103 | 105 / 102 | 106 / 103 | 106 / 105 | 107 / 105 | 106 / 105 | 106 / 105 |
| `video_test2` | 15 / 16 | 15 / 16 | 15 / 16 | 15 / 15 | 15 / 15 | 14 / 15 | 14 / 14 |
| `video_test3` | 45 / 41 | 45 / 45 | 45 / 45 | 44 / 45 | 45 / 45 | 44 / 44 | 45 / 44 |
| `video_test4` (referință ≈48) | 26 / 31 | 37 / 39 | 43 / 46 | 46 / 49 | 48 / 48 | 48 / 47 | 48 / 48 |

Pe `video0`, cu linia mai jos de 0,55 se pierde banda din stânga (iese lateral din imagine); pe `video_test4`, cu linia
mai sus de 0,70 mașinile se acoperă. De aici regula din program: se numără la toate cele 7 linii și se folosește cea cu cele mai multe
vehicule. Cu linia aleasă astfel, cele două modele numără practic la fel (68/68; 107/105; 15/16; 45/45; 48/49).
Secțiunile 2–3 sunt măsurate cu linia fixă la 0,50, înainte de această regulă.

## 3. Pragul de încredere al detecțiilor transmise urmăritorului (YOLO26s)

| Prag | `video0`: corecte / false | `video_test2` |
|---|---|---|
| 0,5 (actual) | 68/68 / 0 | 15/16 |
| 0,25 | 68/68 / **2** | 16/16 |
| 0,1 | 68/68 / **3** | 16/16 |
| YOLO11s, 0,1 + `bytetrack_strict.yaml` | 68/68 / 0 | 16/16 |

Ultralytics configurează ByteTrack cu `track_high_thresh` 0,25, `track_low_thresh` 0,1 și `new_track_thresh` 0,25.
Cu pragul de 0,5 aplicat înaintea urmăritorului, a doua etapă de asociere (detecțiile slabe) nu primește nimic.
Numărările false de la pragurile mici sunt piste duplicate scurte, pornite din detecții slabe ale unor vehicule
deja urmărite. În `bytetrack_strict.yaml`, detecțiile slabe pot doar prelungi o pistă existentă, iar o pistă nouă
pornește numai de la încredere ≥ 0,6.

## 4. Identificarea automată a marcajului cu mașinile detectate de alt model

| Clip | Mașini YOLO11s | Mașini YOLO26s | Marcaj identificat (ambele) | Perioada estimată din mașini |
|---|---|---|---|---|
| `video0` | 382 | 402 | Germania 6 + 12 m, încredere ridicată (corect) | 25,1 m / 24,1 m (real 18 m) |
| `video_test1` | 467 | 499 | Marea Britanie 2 + 7 m, încredere ridicată (corect) | 9,8 m / 9,4 m (real 9 m) |
