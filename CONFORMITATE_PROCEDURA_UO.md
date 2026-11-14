# Conformitatea proiectului cu procedura UO (SEAQ_PO_Pr.MA_06)

Procedura operațională privind elaborarea lucrării de finalizare a studiilor, Universitatea din Oradea, aprobată în
Senat la 16.12.2019. Fișierul acesta ține evidența a ceea ce cere procedura și a stării proiectului față de fiecare cerință.
Stare la 14.11.2026.

## 1. Ce trebuie hotărât de tine sau de coordonator (nu se poate rezolva din cod)

| Nr. | Problemă | De ce contează | Ce e de făcut |
|---|---|---|---|
| 1 | **Asistența AI la scrierea codului** | Anexa 10: declari pe proprie răspundere că lucrarea e scrisă de tine „fără nici un ajutor neautorizat”. Procedura (2019) nu spune nimic despre AI, deci dacă e „autorizat” hotărăște coordonatorul / facultatea. Folosirea e vizibilă public: un commit din depozitul GitHub are „Co-Authored-By: Claude”, iar `CLAUDE.md` e urmărit în git | Întrebi coordonatorul, în scris, dacă și în ce condiții e permisă. Dacă da, o menționezi explicit în lucrare (ce unealtă, la ce a fost folosită). Trebuie să poți explica la susținere fiecare parte din cod |
| 2 | **Sursa videoclipurilor** (`video0`, `video_test1`–`4`) | Anexa 10: nicio sursă folosită fără respectarea drepturilor de autor. În proiect nu e notat de unde provin și sub ce licență | Completezi tabelul din secțiunea 4. Clipurile fără sursă și licență clare nu ar trebui să apară în lucrare |
| 3 | **Plăcuțele de înmatriculare din `video_test4`** | Numerele citite sunt ale unor mașini reale; apar în videoclipul adnotat, în CSV și ar apărea în figurile lucrării | De discutat cu coordonatorul dacă în figuri se maschează parțial numerele |
| 4 | **Licența depozitului public** | Biblioteca Ultralytics (modelul YOLO) are licența AGPL-3.0, care pune condiții codului distribuit împreună cu ea. Depozitul GitHub nu are fișier de licență | De hotărât dacă adaugi o licență (AGPL-3.0 e varianta compatibilă). Nu am adăugat nimic |

## 2. Cerințe de redactare (pentru lucrarea scrisă)

Lucrarea nu se generează din acest folder; lista e aici ca să poată fi verificată la final.

- Denumire: **Proiect de diplomă** (domeniul Științe inginerești).
- Titlu: recomandat cel mult zece cuvinte.
- Părți obligatorii, în ordine: copertă (Anexa 1), pagină de gardă (Anexa 2), cuprins (Anexa 3), introducere, parte
  teoretică, parte practică, concluzii, bibliografie (Anexa 9), anexe, declarație de autenticitate (Anexa 10).
- Introducerea: motivația alegerii temei, importanța ei, obiectivele.
- Partea teoretică: abordare comparativă și critică, nu strict descriptivă; date actuale; să susțină partea practică.
- **Contribuția proprie: cel puțin o treime din lucrare.**
- Minimum 30 de pagini, fără anexe.
- Pagină A4, margini 2,5 cm; Times New Roman 12 pt, la 1,5 rânduri, aliniat stânga-dreapta.
- Titluri de capitol: numerotate, 14 pt, centrate, aldine. Titluri de subcapitol: numerotate, 12 pt, aliniate la stânga.
- Tabele, figuri, grafice: în text, numerotate și cu titlu; referire „figura nr. i.j” / „tabelul nr. i.j” (i = capitolul).
- Bibliografie: tot ce e listat trebuie citat în text, sub forma [n]; formatul intrărilor e în Anexa 9.
- Predare: tipărit și electronic, la coordonator; coordonatorul o depune cu cel puțin 5 zile lucrătoare înainte de susținere.

## 3. Componente ale altor autori folosite în proiect

Anexa 10 cere ca nicio parte a lucrării să nu conțină aplicații publicate de alți autori nedeclarate. Tot ce e mai jos
trebuie prezentat în lucrare ca fiind preluat, cu trimitere bibliografică. Licențele sunt citite din pachetele instalate.

| Componentă | La ce e folosită | Licență |
|---|---|---|
| Ultralytics 8.4 (modelul `yolo26s.pt`, antrenat pe COCO; implementarea ByteTrack) | detecția și urmărirea vehiculelor | AGPL-3.0 |
| fast-alpr 0.4, fast-plate-ocr 1.1, open-image-models 0.6 (cu modelele lor ONNX) | detecția plăcuțelor și citirea textului | MIT |
| OpenCV 5.0 | citire/scriere video, morfologie, flux optic Lucas-Kanade, omografii | Apache 2.0 |
| SciPy 1.18 | optimizarea prin cele mai mici pătrate (`least_squares`), filtre | BSD |
| NumPy 2.5 | calcul numeric | BSD |
| PyTorch 2.15, ONNX Runtime 1.32 | rularea rețelelor neuronale | BSD / Apache / MIT |
| CustomTkinter 5.2, Pillow 13 | interfața grafică | MIT / MIT-CMU |
| lap 0.5 | asocierea din ByteTrack | BSD-2-Clause |

Rețelele neuronale **nu** sunt antrenate în acest proiect: sunt modele gata antrenate, folosite ca atare.

Trimiteri bibliografice candidate (scrise din memorie: **de verificat una câte una**, cu pagini și an, înainte de a intra în lucrare):

- Zhang, Y., Sun, P., Jiang, Y. ș.a.: ByteTrack: Multi-Object Tracking by Associating Every Detection Box, ECCV, 2022.
- Jocher, G., Qiu, J., Chaurasia, A.: Ultralytics YOLO (software), https://github.com/ultralytics/ultralytics
- Lin, T.-Y. ș.a.: Microsoft COCO: Common Objects in Context, ECCV, 2014.
- Lucas, B. D., Kanade, T.: An Iterative Image Registration Technique with an Application to Stereo Vision, IJCAI, 1981.
- Shi, J., Tomasi, C.: Good Features to Track, CVPR, 1994.
- Fischler, M. A., Bolles, R. C.: Random Sample Consensus, Communications of the ACM, 24 (6), 1981.
- Virtanen, P. ș.a.: SciPy 1.0, Nature Methods, 17, 2020.
- Harris, C. R. ș.a.: Array programming with NumPy, Nature, 585, 2020.
- Bradski, G.: The OpenCV Library, Dr. Dobb's Journal, 2000.
- fast-alpr / fast-plate-ocr / open-image-models (software), https://github.com/ankandrew

## 4. Videoclipurile folosite (de completat)

| Fișier | Conținut | Sursă (adresă, autor) | Licență / drept de folosire | Data accesării |
|---|---|---|---|---|
| `video0.mp4` | autostradă, două sensuri (marcaj german) | ? | ? | ? |
| `video_test1.mp4` | autostradă, două sensuri (marcaj britanic) | ? | ? | ? |
| `video_test2.mp4` | autostradă, un sens, văzută de sus | ? | ? | ? |
| `video_test3.mp4` | drum urban în curbă | ? | ? | ? |
| `video_test4.mp4` | drum cu trafic dens, plăcuțe britanice lizibile | ? | ? | ? |

## 5. Contribuția proprie (ce se poate revendica)

Scris în acest proiect, peste componentele din secțiunea 3:

- numărarea la linie cu sensul dedus din mișcare; identificarea vehiculelor roșii (prag HSV);
- calibrarea automată din marcajele discontinue: mai multe linii pe drum drept; o singură linie, cu liniuțe alungite sau scurte;
- identificarea tipului de marcaj din raportul liniuță/perioadă și din dimensiunile aparente ale mașinilor;
- stabilizarea (compensarea mișcării camerei) și estimarea vitezei prin regresie pe poziția punctului de contact cu solul;
- votul pe caractere peste citirile repetate ale plăcuțelor;
- interfața grafică, fereastra de calibrare, tabelul CSV;
- testul automat pe `video0.mp4`, cu 68 de treceri adnotate manual (`ground_truth.json`).

Ce poți revendica personal depinde de răspunsul la punctul 1 din secțiunea 1.

## 6. Afirmații din proiect care nu au încă o sursă sau o verificare

Într-o lucrare științifică fiecare cifră trebuie să poată fi apărată. Acestea nu pot fi, deocamdată:

| Ce | Unde | Stare |
|---|---|---|
| Dimensiunile marcajelor pe țări (`MARKING_STANDARDS`) | `calibration.py` | scrise din memorie, fără normativ citat; România lipsește |
| Constantele „lățime aparentă 1,95 m”, „flanc vizibil 3,4 m”, „neclaritate 2 px” | `auto_calibration.py` | ajustate pe clipurile `video0` și `video_test1`; nu sunt validate pe date independente |
| Vitezele de pe toate clipurile | rezultate | nu există nicio măsurătoare de referință a vitezei (radar, GPS); se poate susține doar coerența internă |
| Scara pe `video_test2`, `video_test3`, `video_test4` | rezultate | programul însuși o marchează „nesigură”; pe `video_test3` a variat între 6 și 9 m la mici schimbări |
| Numărarea pe `video_test1`–`4` | rezultate | verificată doar vizual, parțial; referință adnotată există doar pentru `video0` |
| Acuratețea LPR | `video_test4` | 5 numere verificate vizual din 19 citite; fără listă de referință |

Singurul rezultat cu referință independentă este numărarea pe `video0.mp4` (34 + 34 din 68, 5 vehicule roșii din 5).
