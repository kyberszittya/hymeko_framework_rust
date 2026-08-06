# KIFÜ apology + remediation email draft (Hungarian, formal)

**Verzió: 2 (frissítve a 3. warning után, 20:16)**

**Címzett:** `oreply-hpc@dkf.hu`
**CC (opcionális, ha tudjuk):** a `pr_szevis` allocation PI-je
**Tárgy:** `pr_szhc` — alacsony hatékonyság HSiKAN array jobokon
(2026-06-04) — gyökérok-elemzés, intézkedés, és proaktív álláspont

---

Tisztelt KIFÜ HPC csoport!

Köszönöm a 2026-06-04-én küldött **három** erőforrás-hatékonysági
figyelmeztetést (12:16, ~19:30, 20:16) a 13885808 + 13885809 +
13885810 SLURM array jobokra (összesen ~108 alacsony TimeEff-ű
feladat). A figyelmeztetéseket komolyan veszem; a gyökérokot már
azonosítottam, az intézkedést bevezettem, és **a felhasználói
várólistámat üresre állítottam, amíg a javított submitter
verifikálva nincs**.

## 1. A gyökérok (egy konkrét script)

A három array job egyetlen submitter scriptből származott
(`docs/komondor_setup/submit_hsikan_edge_cr_array.sh`, "v1"),
amely **egységes `--time=02:30:00`** értéket állított minden
cellára. Ez az időkorlát a HSiKAN-edge_cr SOTA konfiguráció
**legrosszabb esetére** (Epinions hideg-cache cycle-enumeráció,
~2 óra 28 perc) volt méretezve, biztonsági ráhagyással. A 40
cellás grid azonban nem-uniform:

| cellaosztály | cellaszám | tényleges futás | TimeEff a 2:30:00 limit alatt |
|---|---|---|---|
| Bitcoin Alpha + OTC | 20 | ~30 s | **0.3 %** |
| Slashdot real (meleg-cache) | 5 | ~35 s | **0.4 %** |
| Slashdot shuffle | 5 | ~10-11 perc | ~7 % |
| Epinions real + shuffle (hideg-cache) | 10 | ~2 h 27 m | ~98 % |

A 25 kis cella elnyomta az átlagot a 0.3-0.8 % tartományba, és
így triggerelte a figyelmeztetést mind a három array job
befejeződésekor (ahogyan a 4 órás reportseff ablak haladt).

## 2. Az intézkedés (már megtörtént)

1. **`scancel 13888159`** — az utolsó még pending feladatomat
   visszavontam. **A queue 2026-06-04 19:55-óta üres**, és a
   javított submitter-verifikáció előtt nem küldök új jobot.

2. **v2 submitter ship** — új submitter script
   (`docs/komondor_setup/submit_hsikan_edge_cr_array_v2.sh`) a
   gridet **három időosztályra bontja**:

   | osztály | cellaszám | --time | projektált TimeEff |
   |---|---|---|---|
   | TINY (BA + OTC + Slashdot-real) | 25 | 00:05:00 | ~10 % (median 30 s) |
   | MEDIUM (Slashdot-shuffle) | 5 | 00:30:00 | ~33 % (median 10 min) |
   | LONG (Epinions real + shuffle) | 10 | 04:00:00 | ~62 % (median 2.5 h) |

   Mindhárom osztály a `pr_szevis` allocation alatt 10 % feletti
   TimeEff-fel zárul; egyik sem éri el az automata
   figyelmeztetési küszöböt.

3. **Diagnosztikai dokumentum** — a hiba teljes elemzése +
   adatlevezetés:
   `reports/2026-06-04-kifu-resource-eff-response.md` a projekt
   repóban. Tartalmazza a per-cella wall hisztogramot, a
   cold/warm cache aszimmetriát, és a remedy specifikációt.

4. **Audit metrics tooling kibővítve** —
   `scripts/komondor_audit_metrics.py` és
   `scripts/komondor_morning_pull.sh` mostantól önként lekérdezik
   a `reportseff` adatokat minden array-pull után, így a következő
   submission előtt **mi magunk** látjuk az efficiency-t, és nem
   a cluster automata rendszerének kell ezt jeleznie.

## 3. Mi várható a következő 4 órás ablakokban

Megjegyzem, hogy a 13885810 (K=5) array további cellái fokozatosan
fognak kicsúszni a 4 órás reportseff ablakából. Lehetséges,
hogy még **1-2 további figyelmeztetést** kapok a KIFÜ rendszerétől
ezekről a már lefutott cellákról, mielőtt a teljes K=5 array
kicsúszik az ablakból (várhatóan 2026-06-04 24:00 körül). **Ezek
visszamenőlegesen ugyanannak a v1 submitter-fejlesztési hibának a
követelései**, nem új jobok; **nem indítok új jobot**, ami további
warningot okozhatna, amíg a v2 verifikálva nincs.

## 4. Mit nem fogok tenni a v2 fix nélkül

A `pr_szevis` allocation alatt **nem indítok új array vagy
sequence jobot**, amíg a v2 submitter sizing-jét **egyetlen, kis
dry-run array-en** (4-cellás Epinions shuffle re-run a v2 LONG
osztály alatt) nem verifikálom. Ez a dry-run csak akkor indul,
ha:

- A 13885810 K=5 array teljes egészében kicsúszott a 4 órás
  reportseff ablakából (várhatóan 2026-06-04 23:30-24:00 körül).
- Egyértelmű felhasználói (saját) jóváhagyás megtörtént.

A dry-run várt TimeEff > 60 %; az eredményt jelentem.

## 5. Miért dolgozom ezen az allocation alatt

Az allocation alatt futó HSiKAN-edge_cr benchmark a 2026-os
Nature Communications-be benyújtásra előkészített cikkemhez
tartozik (`paper/nature_comm_v1/main.tex`). A 2026-06-04-i
Slashdot 5-seed eredmény (AUC `0.9058 ± .0033`) a már publikált
SOTA-t (`0.9067 ± .0029`) reprodukálja a Komondor A100-on; az
Epinions 5-seed eredmény (`0.8829 ± .0128`) új SOTA-jelölt. A
ti audit gridje nélkül az 5-seed paired protokoll nem
futtatható; a v1 fix nélkül nem tudom etikusan tovább használni
az allocation-t.

A `pr_szhc` felhasználói viselkedését — beleértve a hibákat — a
projekt PI-jének is felülvizsgálhatóvá teszem (a repó snapshot
publikus, a benchmark-eredményeket aláírom). A KIFÜ csapatának
bármilyen formában szívesen biztosítok további diagnosztikát,
ha az segít a probléma megelőzésében más felhasználóknál is.

Köszönöm a türelmét és a figyelmeztetéseket. A KIFÜ automata
visszacsatolása nélkül a hibát órákkal később vettem volna észre.

Tisztelettel,

Hajdu Csaba (`pr_szhc`)
Széchenyi István Egyetem
HSiKAN / `pr_szevis` allocation
`gemeauxrapace@gmail.com`

---

**Hivatkozott artifaktok (a `pr_szhc` repójában, audit célra
hozzáférhetőek):**

- `docs/komondor_setup/submit_hsikan_edge_cr_array.sh` (régi v1, hibás)
- `docs/komondor_setup/submit_hsikan_edge_cr_array_v2.sh` (új v2, javított)
- `reports/2026-06-04-kifu-resource-eff-response.md` (részletes elemzés)
- `scripts/komondor_audit_metrics.py` (önaudit script)
- `scripts/komondor_morning_pull.sh` (one-shot pull + audit)
