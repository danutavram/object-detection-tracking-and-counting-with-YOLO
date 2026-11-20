# Sistem de detecție, numărare și estimare a vitezei autovehiculelor

Proiect de diplomă – Avram Nicolae Dan, Universitatea din Oradea (RST), coordonator Prof. univ. dr. ing. Cristian Grava.
Limba de lucru cu utilizatorul: română.

Lucrarea de diplomă: `../Proiect_diploma_Avram_Nicolae_Dan.docx` (se modifică din aplicația Claude desktop, nu de aici).
`../Avram_Nicolae_Dan_Proiect_Prelucrarea_si_analiza_imaginilor.docx` este proiectul vechi de semestru, nu lucrarea de diplomă.
Lucrarea (stare 11.11.2026) descrie „versiunea 1” (YOLO11s, urmăritor pe centroid, două linii de viteză, omografie fixă) și o
„versiune 2” (calibrare pe 42 de puncte doar pentru `video0`, ByteTrack, regresie Y(t)). Codul actual este o singură versiune,
mai avansată: calibrare automată pe orice clip, stabilizare, regresie 2D, LPR, 5 clipuri. Timpii din lucrare au fost măsurați
de o sesiune anterioară pe o mașină cloud (Intel Xeon 2,1 GHz, 2 nuclee), nu pe calculatorul utilizatorului.

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

6. **Lucrarea analizează doar versiunea actuală a codului** (decizie 20.11.2026). Fără „versiunea 1” / „versiunea 2”, fără
   comparații între versiuni, fără codul vechi în anexe; anexa de cod = codul actual. Toate rezultatele din cap. V se refac
   cu codul actual. „Ne interesează ce e acum, nu ce a fost”: fără comparații înainte/după (procesor față de placa video,
   înainte/după o corecție); timpii din lucrare sunt cei de pe placa video. **Nici comparații între modele**: celelalte
   modele YOLO doar se menționează că există, cu minimum de detalii; lucrarea descrie modelul folosit.
   `experimente/` e intern (sprijină alegerea modelului și a pragului), nu intră în lucrare.
   Atenție la procedura UO: partea teoretică trebuie să aibă „abordare comparativă și critică” – comparația se poate face
   între metode (tehnologii de detecție, detectoare într-o etapă / două etape, metode de urmărire), nu între variante YOLO.
   Ce trebuie rescris în lucrare: vezi „Probleme cunoscute” 8.

## Structura

| Fișier | Rol |
|---|---|
| `main.py` | rulare fără interfață pe `video0.mp4` |
| `gui.py` | interfața CustomTkinter + fereastra de calibrare (`CalibrationWindow`) |
| `main_tracker.py` | bucla principală: cadru → detecție+urmărire → stabilizare → numărare/viteză/roșu → adnotare → video + CSV |
| `model_loader.py` | încarcă modelul (`MODEL_PATH = "yolo26s.pt"`) |
| `vehicle_detector.py` | `model.track(..., tracker="bytetrack.yaml", classes=[2,3,5,7], conf=0.5)` |
| `tracker.py` | clasa `Tracker`: numărare la 7 linii (0,50–0,80 din înălțime), se folosește cea cu cele mai multe vehicule; sens din direcția de mișcare, roșu, viteze |
| `auto_calibration.py` | geometria din marcaje: mai multe linii discontinue pe drum drept (fundal → punct de fugă → raze → liniuțe periodice → model de cameră) sau o singură linie, eventual în curbă (lanț de liniuțe); estimarea scării din casetele mașinilor |
| `calibration.py` | clasa `Calibration`, tabelul `MARKING_STANDARDS`, identificarea automată a marcajului, calibrare automată/manuală, salvare `<video>.calib.json`, `Stabilizer`, grila de verificare, CLI |
| `speed_estimator.py` | punct de contact cu solul → metri → regresie liniară X(t), Y(t) (2D, pentru drum în curbă); fără cadrele în care vehiculul e acoperit de cel din față; afișare doar când estimarea e stabilă |
| `plate_reader.py` | LPR cu `fast-alpr` (detector de plăcuțe + OCR, ONNX): fiecare vehicul e citit în mai multe cadre, numărul final prin vot pe caractere; apare în etichetă și în CSV |
| `red_vehicle_counter.py` | roșu în Lab, față de dominanta de culoare a cadrului, pe partea vizibilă a casetei (proporție ≥ 0,17) |
| `video_manager.py` | deschidere video / VideoWriter |
| `test_sistem.py`, `ground_truth.json` | test automat pe `video0.mp4`: calibrare automată + 68 de treceri adnotate manual |
| `experimente/` | măsurători reproductibile, interne (nu fac parte din aplicație și nu intră în lucrare): `compara_modele.py`, `timp_detectie.py`, `bytetrack_strict.yaml`; rezultatele în `REZULTATE_MODELE.md` și `*.json` |

Contractul cu interfața: `real_time_detection_with_callback(video_path, output_path, frame_callback=None, stop_event=None,
read_plates=True)`; `frame_callback(frame, stats)` cu cheile `entry_count` (se apropie), `exit_count` (se depărtează),
`red_vehicle_count`, `max_speed`, `plates_read`, `plates` (listă de (id, număr, viteză), doar vehiculele numărate),
`plates_status` (None = citirea merge; „oprita”; „indisponibila (motiv)”), `line_ratio` (înălțimea liniei alese),
`frame_number`, `total_frames`, `calibrated`. Rezultatul final are aceleași chei plus `table_path`, `counted`, `speeds` etc.

Interfața (`gui.py`, 20.11.2026): texte în română cu diacritice; comutator pentru citirea numerelor; mesaj când citirea
nu e disponibilă; afișează linia de numărare aleasă; la final rezumat + buton „Deschide folderul cu rezultatele”;
închiderea ferestrei în timpul procesării oprește întâi procesarea (videoclipul de ieșire rămâne lizibil – verificat).
Fundalul pentru stabilizare (≈15 s: citirea a 60 de cadre + mediana) se calculează o dată pe sesiune pentru fiecare
videoclip (`calibration.video_background`) și începe imediat ce se alege videoclipul. Tipurile de vehicule se afișează
în română (`vehicle_detector.CLASS_NAMES`); detecțiile „train” (în clipuri nu e niciun tren: modelul confundă un vehicul
mare) apar ca „vehicul mare”. Testată cu acțiuni simulate și capturi de ecran: `video_test4` cu citirea numerelor până
la final, `video_test2` fără ea, închisă în timpul procesării, fereastra de calibrare pe `video_test3` (fără salvare).

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
  ≈5–35 % (9,4 m față de 9 pe `video_test1`; 24,1 față de 18 pe `video0`, `test_sistem.py` 20.11.2026), deci singură nu ajunge: de aceea se combină cu
  raportul liniuță/perioadă și se alege un standard din tabel. Valorile din tabel sunt din memorie, neverificate în normative.
  Utilizatorul poate impune un tip din listă; dacă raportul lui nu corespunde cu imaginea, interfața avertizează.
- **Calibrare manuală** (interfață): 4 click-uri pe un dreptunghi de pe drum + lățime și lungime în metri.
- **Stabilizare**: deplasarea camerei față de fundal se măsoară la fiecare 5 cadre (Lucas-Kanade) și se scade din poziția vehiculului.
  În `video0.mp4` camera alunecă ≈21 px în primele 3 secunde; fără stabilizare două vehicule aveau erori de până la 15 %.
- **Intrarea în cadru**: cât timp caseta atinge fâșiile de la margine (7 % jos, 2 % sus) pozițiile nu se folosesc; viteza se
  afișează după ≥1 s, ≥10 puncte și eroare standard a pantei ≤3 %. Înainte, prima valoare afișată era în medie cu 12 % prea mică.
- Punctul folosit este mijlocul laturii de jos a casetei (contactul cu solul), nu centrul casetei.
- **Vehicul acoperit de cel din față** (coloană deasă): caseta cuprinde doar partea vizibilă, latura ei de jos nu e pe sol
  și coboară mai repede decât vehiculul pe măsură ce acesta iese de după celălalt. Pe `video_test4` asta dădea 98–100 km/h
  într-o coloană de ≈45 km/h (ID 97: înălțimea casetei 62–244 px față de ≈290 px a vehiculului întreg). Cadrele în care
  punctul de contact cade în caseta unui vehicul mai apropiat **și** caseta e mai scurtă decât 0,85 × percentila 90 a
  raportului înălțime/lățime pe aceeași pistă nu se folosesc. Viteza se recalculează la fiecare cadru și dispare dacă
  estimarea nu mai e stabilă. Verificat: `video_test4` 39–100 → 38–63 km/h (44 din 48 cu viteză); `video0` neschimbat
  (68/68, 74–187); `video_test1` 105/107; `video_test3` 40/45 (4 vehicule din trafic oprit-pornit rămân fără viteză).
  Doar criteriul „punct acoperit” (fără forma casetei) elimina 88 % din cadre pe `video_test4` – prea strict.

## Culoarea roșie

Proporția pixelilor roșii din partea vizibilă a casetei, în momentul traversării liniei, în spațiul CIE Lab, față de
dominanta de culoare a cadrului (mediana a*, b*): croma relativă ≥ 10, unghiul nuanței între −20° și +30°, L* ≥ 20;
vehiculul e roșu dacă proporția ≥ 0,17. Partea acoperită de casetele vehiculelor mai apropiate nu se folosește.
Motivul: pe `video_test1` (apus, tentă albastră) mașinile roșii apar cu nuanța HSV 150–170 (în afara intervalului vechi
0–15 / 165–180), iar autocamionul TNT portocaliu cade în 0–5; regula veche (HSV, proporție ≥ 0,13 pe toată caseta)
dădea 4 din 7 roșii + TNT. Clasificare manuală pe planșe cu toate vehiculele numărate (20.11.2026):

| Clip | Roșii reale | Regula veche | Regula nouă |
|---|---|---|---|
| `video0` | 5 | 5 corecte | 5 corecte |
| `video_test1` | 7 (+1 vișinie închisă, ambiguă) | 4 corecte + TNT | 7 corecte + vișinia |
| `video_test2` | 2 | 2 corecte | 2 corecte |
| `video_test3` | 5 | 5 corecte | 5 corecte |
| `video_test4` | 1 (acoperită pe jumătate de o mașină albă) | 0 | 1 corectă |

Pragurile sunt alese pe aceleași clipuri (roșiile ≥ 0,22, celelalte ≤ 0,12, inclusiv mașini închise cu stopurile
aprinse), deci nu sunt validate pe date independente.

## Cum funcționează numărarea

Un vehicul se numără când centrul casetei lui traversează o linie orizontală; sensul e dat de direcția traversării.
Se numără în paralel la 7 linii (0,50; 0,55; …; 0,80 din înălțime) și se folosește linia cu cele mai multe vehicule
(la egalitate, cea mai apropiată de mijloc). Motivul (20.11.2026): nicio înălțime fixă nu merge pe toate clipurile.
- `video_test4` (coloană filmată cu teleobiectivul): la 0,50 mașinile se acoperă, urmăritorul le pierde chiar la linie
  și programul număra 26 (YOLO26s) / 31 (YOLO11s); de la 0,70 în jos ambele modele numără 48. Referința: 48 ± 5,
  numărate manual pe imaginea-timp de la 0,88 din înălțime (`experimente/imagine_timp.py`; 12 + 10 + 14 + 12 pe cele 4
  imagini; nesigur în porțiunile 18–21 s și 32–42 s, banda din stânga).
- `video0`: linia de jos ratează banda din stânga, care iese prin marginea laterală (0,60: 31 + 34; 0,70: 27 + 32).
  Linia aleasă rămâne 0,50: 34 + 34, exact.
Numărul afișat nu scade niciodată (fiecare linie numără crescător), dar linia aleasă se poate schimba în primele secunde.

## Rezultate (Windows, placa video, YOLO26s, numărare cu linia aleasă automat, 20.11.2026)

| Clip | Marcaj identificat automat | Calibrare | Linia | Numărare | Viteze |
|---|---|---|---|---|---|
| `video0.mp4` | Germania, autostradă 6+12 m (corect), încredere ridicată | 4 linii, eroare 0,45 m | 0,50 | 34 + 34, exact față de referință; 5 roșii din 5 | 68 din 68; 74–187 km/h |
| `video_test1.mp4` | Marea Britanie 2+7 m (corect), încredere ridicată | 4 linii, 0,28 m | 0,70 | 55 + 52; 8 roșii = cele 7 reale + 1 vișinie | 105 din 107; 30–113 km/h |
| `video_test2.mp4` | „6 m + 9 m”, **încredere scăzută** (standard real necunoscut; din mașini 14,3 m) | 2 linii, 0,19 m | 0,50 | 16 din 16 (autoutilitara „train” inclusă); 2 roșii din 2 | 16 din 16; 59–95 km/h |
| `video_test3.mp4` | „Polonia urban 2+4 m”, **încredere scăzută** (din mașini 6,6 m). Instabil: la mici schimbări ale măsurării raportului a ieșit pe rând 7,5 / 9 / 6 m | 1 linie (lanț de 5 liniuțe), 0,43 m | 0,50 | 45 (verificat pe imaginea-timp: 44 corecte, 1 dublă); 5 roșii din 5 | 40 din 45; 14–84 km/h (trafic oprit-pornit) |
| `video_test4.mp4` | Marea Britanie 2+7 m, **încredere scăzută** (ales doar din raportul liniuță/perioadă 0,21; mașinile nu determină scara la teleobiectiv). Coerent cu plăcuțele britanice | 1 linie (24 de liniuțe scurte, `straight_dash_line`), 0,82 m; cameră la ≈6 m | 0,70 | 48 (referință manuală 48 ± 5); 1 roșie din 1 | 44 din 48; 38–63 km/h |

Tirul cu 180 km/h de pe `video_test1` venea din calibrarea făcută în interfață cu marcajul implicit german (perioadă 18 m în loc de 9 m).

Verificări din 20.11.2026:
- **Modelul: YOLO26s** (rămâne). Cu linia aleasă automat, YOLO26s și YOLO11s numără la fel: `video0` 68/68 amândouă;
  `video_test1` 107 / 105; `video_test2` 15 / 16; `video_test3` 45 / 45; `video_test4` 48 / 49. Verificat în imagini:
  cele 2 vehicule în plus ale YOLO26s pe `video_test1` (20,34 s – mașină roșie; 35,29 s – autoutilitară albastră) sunt
  reale și ratate de YOLO11s (nu duble, cum era notat înainte); pe `video_test3`, la linia 0,50, YOLO11s ratează 3
  vehicule reale; pe `video_test2`, YOLO26s clasifică autoutilitara de la 25,07 s drept „train” – rezolvat prin includerea
  clasei „train” (pe celelalte 4 clipuri nu schimbă nimic). La egalitate s-a păstrat YOLO26s: cel mai nou, precizie COCO
  publicată mai mare (48,6 față de 47,0), deja în cod. Identificarea marcajului dă același rezultat cu mașinile ambelor modele.
  YOLO26m (pe placa video doar +1 ms/cadru): 68/68; 107; 16; 48 pe `video_test3` (≈3 peste referință); 48; ratează câte
  o mașină roșie pe `video_test3` și `video_test4` (pragurile de roșu sunt alese pe casetele YOLO26s) – nu e mai bun.
- **Prag de încredere**: 0,25 / 0,1 recuperează autoutilitara, dar produc 2 / 3 numărări duble pe `video0`.
  `bytetrack_strict.yaml` (detecții de la 0,1, pistă nouă doar de la 0,6) + YOLO11s: 68/68 fără false și 16/16.
- **Roșii**: vezi „Culoarea roșie” (regula veche, cu linia 0,50, dădea pe `video_test1` 13, din care 5 false de la
  stopuri aprinse și TNT-ul portocaliu).
- **Vitezele mici pe `video_test1`** (se depărtează: mediana 37 km/h; se apropie: 92 km/h) sunt reale, nu o eroare de
  calibrare: la aceleași rânduri din imagine vehiculele se mișcă cu 2,5 față de 5,4 px/cadru (același raport 2,2).
- Clipurile au cadre la intervale practic egale (abatere maximă sub un cadru față de n/fps).

## Probleme cunoscute / de făcut

1. ~~LPR oprește programul fără internet~~ – rezolvat 20.11.2026: `create_plate_reader()` prinde orice excepție, scrie
   un mesaj în consolă și programul continuă fără LPR (verificat prin simulare cu `URLError`). Interfața nu arată mesajul.
2. ~~Numărul de plăcuțe diferă între imagine și interfață~~ – rezolvat: peste tot (imagine, interfață, CSV) se raportează
   doar plăcuțele vehiculelor numărate (`plates` din callback și `plates_read`).
3. ~~Viteze greșite în coloană deasă~~ – rezolvat: vezi „Cum funcționează viteza”, vehicul acoperit.
4. **Pragul 0,5 aplicat înainte de ByteTrack** dezactivează a doua etapă de asociere a ByteTrack (praguri implicite
   0,25 / 0,1). `bytetrack_strict.yaml` testat pe toate 5 clipurile (linia fixă 0,50, `REZULTATE_MODELE.md`): cu YOLO26s
   numără mai puțin pe `video_test3` (43 față de 45) și `video_test4` (20 față de 26), deci nu e mai bun; rămâne 0,5.
   În lucrare trebuie spus că a doua etapă de asociere a ByteTrack nu primește detecții (descriere corectă, nu „ByteTrack complet”).
5. **Rulare pe placa video** (RTX 3060 Ti, PyTorch 2.15.0+cu132, vezi „Mediul Python”); timpii actuali sunt în
   `experimente/REZULTATE_MODELE.md`. `test_sistem.py` pe `video0`: 85 s (80 ms/cadru), 68/68, 5 roșii, 68 viteze. LPR
   (ONNX Runtime) rulează pe procesor, dar pe `video0` nu încetinește (93 s fără). Restul timpului (pregătirea fundalului
   ≈15 s, citire/scriere video, ByteTrack, stabilizare) nu e măsurat separat. Nu e timp real (clipul are 42,4 s).
   Timpii din tabelul „Rezultate” de mai sus și din lucrare sunt vechi (procesor); pentru lucrare contează doar cei noi.
6. **Fișierul CSV cu numere de înmatriculare e public pe GitHub** (`video_test4_output.csv`, commit 5544bb9, împins pe
   `origin/main`). 20.11.2026: ieșirile generate (`*_output.csv/.mp4`, `output_test.*`) au fost șterse din folder, iar
   CSV-urile adăugate în `.gitignore`; ștergerea nu e încă în commit. Și după commit fișierul rămâne în istoricul de pe
   GitHub; scoaterea din istoric cere rescrierea istoricului și push forțat (decizia utilizatorului). `*.calib.jpg`
   rămân în git: sunt verificarea vizuală a calibrării și nu se pot reface fără a recalibra.
7. ~~Roșu: alarme false la apus~~ – rezolvat: vezi „Culoarea roșie” (toate cele 5 clipuri corecte, praguri alese pe ele).
8. **Lucrarea** (regula 6: doar versiunea actuală): timpii și comparațiile din ea au fost măsurați pe mașina cloud, cu
   codul vechi; de refăcut pe calculatorul utilizatorului cu `experimente/` și `test_sistem.py`. Etica (cap. V.9) spune
   „nu citește numere de înmatriculare” – nu mai e adevărat. Părți din docx (stare 11.11.2026) legate de v1/v2:
   - Introducere: obiectivul „realizarea unei versiuni îmbunătățite (v2)”; contribuțiile („dezvoltarea versiunii…”);
     obiectivul „compararea variantelor YOLO11 (n, s, m, x) și a algoritmilor de urmărire”.
   - II.4 (tabelul generațiilor YOLO) și II.5 („Arhitectura YOLO11 și variantele sale”): de redus la modelul folosit,
     celelalte doar menționate.
   - III.4: metoda cu două linii „folosită în proiectul inițial” (poate rămâne doar ca metodă din literatură).
   - IV.4–IV.9: descriu modulele v1 (urmăritor pe centroid, prag 50 px, `compute_homographies` pe două sensuri, liniile
     SPEED LINE 1/2, `int(fps)`). Lipsesc modulele actuale: `auto_calibration.py`, `calibration.py` (marcaj identificat
     automat, `Stabilizer`), regresia 2D, `plate_reader.py`, interfața de calibrare.
   - V.4 („Compararea variantelor YOLO11 și a algoritmilor de urmărire”: tabelele 5.2 și 5.3, figura 5.2): iese.
   - V.3–V.7: tabelele 5.1–5.5 și figura 5.5 sunt „versiunea 1” (ex. supraestimarea de 62 % / 10 %); V.8 întreg și
     tabelul 5.6 = comparația v1–v2; V.9 („în versiunea 1…”); Concluzii (punctele despre versiunea inițială/îmbunătățită).
   - Anexa 2 (cod v1), Anexa 3 (`tracker_v2.py`, `main_tracker_v2.py`, `speed_estimator.py` v2) → înlocuite cu codul actual.
   - Lucrarea descrie un singur clip (`video0`, 1060 de cadre); codul actual e testat pe 5.
   Codul v1 mai există doar în istoricul git (commit `e98c5cf`); `all_code.py` și cadrele `frame_*.png` din v1 au fost
   șterse din folder la 20.11.2026.
9. Evaluări lipsă pentru capitolul de rezultate: referință pentru LPR (`video_test4`), pentru numărare pe încă un clip și
   pentru roșii pe `video_test1`; o validare fizică a vitezei (filmare proprie cu viteză cunoscută, GPS/pilot automat).
10. Tabelul `MARKING_STANDARDS` e din memorie. România: AND 616-2022 (Art. 59) trimite la SR 1848-7 pentru dimensiuni;
    CP D.02.In8:2023 (Republica Moldova, Anexa R, „conform SM SR 1848-7”) dă linia discontinuă A: segment/interval 1:3,
    4 m + 12 m peste 60 km/h, 1+3 / 2+6 / 3+9 m sub 60 km/h. De confirmat în SR 1848-7:2015 (bibliotecă / ASRO).
11. Interfața: rezolvate 20.11.2026 textele amestecate, comutatorul LPR, pregătirea lentă la fiecare pornire (vezi
    „Interfața” mai sus). Rămas: calibrarea manuală făcută de un om cu mouse-ul (testată doar cu click-uri simulate);
    prima pornire pe un videoclip tot așteaptă fundalul dacă se apasă PORNEȘTE imediat după alegere.
12. `auto_calibration.py` este scris compact; de făcut mai lizibil fără a schimba comportamentul, dacă intră în anexă.
13. CSV-ul folosește „;” – pe Windows cu setări regionale en-US (ca acest calculator) Excel îl deschide într-o singură coloană;
   cu setări românești, valori ca „8.2” din coloana de timp devin date calendaristice. Niciun CSV nu merge în ambele.
   Soluția propusă: un `.xlsx` alături de CSV (`openpyxl`, MIT); instalarea a fost blocată de permisiuni – decizia utilizatorului.
14. `requirements.txt` nu fixează versiunile. Conflictul `opencv-python-headless` (instalat de `fast-alpr`) nu mai strică
    nimic: codul nu mai folosește funcții de fereastră din OpenCV.

## Mediul Python (Windows)

`python -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt` → `python test_sistem.py` → `python gui.py`.

`requirements.txt` aduce PyTorch pentru procesor. Pentru placa video (instalat pe acest calculator la 20.11.2026, ≈3 GB):
`pip install --force-reinstall --no-deps "torch==2.15.0+cu132" "torchvision==0.29.2+cu132" --index-url https://download.pytorch.org/whl/cu132`.
Ultralytics alege singur placa video când PyTorch o vede (`torch.cuda.is_available()`); codul nu are nimic de setat.
Pentru o rulare doar pe procesor: `set CUDA_VISIBLE_DEVICES=-1` (o valoare goală nu ascunde placa pe Windows).
