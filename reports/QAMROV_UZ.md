# Eksperiment qamrovi: sohalar, kategoriyalar va raqobatchilar

**Sana:** 2026-09-25 · **Manba:** `temoa/registry.py`, `temoa/problems.py`, `experiments/`
· **Natijalar:** `reports/data_30D/` (30D, 12 yurish)

Bu hujjatdagi barcha sonlar koddan hisoblangan yoki o'lchangan. Funksiyalarning
xossalari (shartlanganlik soni, separabellik, lokal minimumlar) taxmin emas —
son-raqamli Gessian va to'liq domen bo'ylab kesimlar orqali **o'lchangan**
(uslub 3-jadval ostida).

---

## 0. Umumiy ko'lam — bir qarashda

| Ko'rsatkich | Qiymat |
|---|---|
| Raqobatchi algoritmlar | **11** (asosiy 9 + 2 nazorat) |
| Test funksiyalari | **12** |
| Masala kategoriyalari | **8** |
| O'lchamlar | **3** (30D, 50D, 100D) |
| Mustaqil yurishlar | **30** |
| Benchmark treklari | **2** (rotated = asosiy, shift = asl to'plam) |
| FES byudjeti | 3000·D (30D→90k, 50D→150k, 100D→300k) |
| **Asosiy tadqiqot yurishlari** | **12 960** |
| Ablatsiya yurishlari | 1 800 (15 konfiguratsiya) |
| Sezgirlik yurishlari | 1 620 (6 sweep, 27 qiymat) |
| **Jami** | **16 380 yurish** |
| Jami funksiya baholashlari | **2.33 milliard** |
| Statistik kataklar (funksiya × o'lcham) | 36 |
| Bitta nazoratga nisbatan juftlik testlari | 396 |

---

## 1. Raqobatchilar — 11 ta, 5 ta oila

TEMOA_V11 quyidagilar bilan kuch sinashadi:

| # | Algoritm | Oila | Yil | Manba | Roli bu tadqiqotda |
|---|---|---|---|---|---|
| 1 | **TEMOA_V10** | gibrid | 2026 | ushbu loyiha | **Asl algoritm** — tuzatish nimaga nisbatan o'lchanadi |
| 2 | **jSO** | adaptiv DE | 2017 | Brest va b., CEC'17 | **Eng qattiq raqobatchi**: TEMOA'ning o'z yadrosi. TEMOA = jSO + 3 operator + eigen + ES dumi |
| 3 | **L-SHADE** | adaptiv DE | 2014 | Tanabe & Fukunaga, CEC'14 | Yadroning ikkinchi manbasi; CEC'14 g'olibi |
| 4 | DE/rand/1/bin | klassik EA | 1997 | Storn & Price | Klassik mos yozuvlar nuqtasi |
| 5 | PSO (constriction) | swarm | 2002 | Clerc & Kennedy | Nashr etilgan turg'un shakl (χ=0.7298, c=2.05) |
| 6 | GWO | swarm | 2014 | Mirjalili va b. | TEMOA operator 1 ning manbasi |
| 7 | WOA | swarm | 2016 | Mirjalili & Lewis | TEMOA operator 2 ning manbasi |
| 8 | SCA | matematik-ilhomli | 2016 | Mirjalili | Trigonometrik yangilanish |
| 9 | HHO | swarm | 2019 | Heidari va b. | TEMOA operator 3 (Levy dive) ning manbasi |
| — | *PSO_orig* | *nazorat* | — | asl kod | Zaiflashtirilgan PSO (divergent konfiguratsiya) |
| — | *HHO_orig* | *nazorat* | — | asl kod | Levy dive olib tashlangan HHO |

**Nega aynan shular.** 6–9 raqamlar tasodifiy tanlanmagan: **GWO, WOA va HHO —
TEMOA o'z operatorlarini qarz olgan manbalar.** Ya'ni gibridni uning tarkibiy
qismlari bilan yuzma-yuz qo'yish — bu gibridlashtirish haqiqatan foyda
berganini ko'rsatishning yagona to'g'ri yo'li. jSO va L-SHADE esa yadroning
o'zi. Asl tadqiqotda oxirgi ikkitasi **yo'q edi**, shuning uchun u
gibridlashtirish hissasini umuman o'lchamagan.

**Nazorat guruhi (PSO_orig, HHO_orig)** — asl tadqiqotdagi zaiflashtirilgan
konfiguratsiyalar. Ular natijalar jadvalining qancha qismi haqiqiy ustunlikdan,
qancha qismi zaif baseline'dan kelganini raqam bilan ko'rsatish uchun ishlaydi.

---

## 2. Masala kategoriyalari — 8 ta

| Kod | Kategoriya | Funksiyalar | Soni | Nimani sinaydi |
|---|---|---|---|---|
| **C1** | Unimodal, yomon shartlangan | BentCigar, RotatedElliptic, Zakharov | 3 | Kovariatsiya bazisini o'rganish, anizotrop qadam |
| **C2** | Tor egri vodiy | Rosenbrock | 1 | Vodiy bo'ylab yo'nalishni kuzatish |
| **C3** | Multimodal, separabel emas | Ackley, Griewank, Levy | 3 | Global qidiruv + aylantirish bilan kurash |
| **C4** | Deceptive multimodal, separabel | Schwefel | 1 | Aldamchi landshaftdan chiqish |
| **C5** | Kompozitsiya / gibrid | Composition | 1 | Aralash landshaft |
| **C6** | Multimodal + shovqin | NoisyRastrigin | 1 | Shovqinda global qidiruv |
| **C7** | Unimodal + shovqin | NoisySphere | 1 | Shovqinda aniq yaqinlashish |
| **C8** | Dinamik (harakatlanuvchi optimum) | DynamicSphere | 1 | Optimumni kuzatish |

> **Muhim cheklov.** 8 ta kategoriyadan **5 tasida atigi 1 ta funksiya bor**
> (C2, C4, C5, C6, C7, C8). Bitta funksiya bo'yicha "bu kategoriyada falon
> algoritm yaxshi" deyish statistik jihatdan asossiz. Kategoriya darajasidagi
> da'vo uchun har birida kamida 4–5 funksiya kerak — bu CEC'2017 (30 funksiya)
> qo'shilishining asosiy sababi (§7).

---

## 3. 12 funksiya — o'lchangan xossalar

| Funksiya | Kat. | cond(H) optimumda | Off-diagonal % | Separabel? | Lokal min (≥) | Aylantirilgan | Chegaralar |
|---|---|---|---|---|---|---|---|
| BentCigar | C1 | **1.0e+06** | 29.9% | yo'q | 1 | ha | [−100, 100] |
| RotatedElliptic | C1 | **1.0e+06** | 86.5% | yo'q | 2 | ha | [−100, 100] |
| Zakharov | C1 | 1.0e+02 | 89.9% | yo'q | 2 | ha | [−10, 10] |
| Rosenbrock | C2 | 1.9e+02 | 65.1% | yo'q | 1 | ha | [−30, 30] |
| Ackley | C3 | 2.3e+00 | 85.1% | yo'q | **73** | ha | [−32, 32] |
| Griewank | C3 | 1.3e+01 | 77.7% | yo'q | 9 | ha | [−600, 600] |
| Levy | C3 | 1.9e+01 | 85.2% | yo'q | 10 | ha | [−10, 10] |
| Schwefel | C4 | 1.0e+00 | **0.0%** | **ha** | 10 | yo'q (faqat sign-flip) | [−500, 500] |
| Composition | C5 | 1.1e+01 | 75.6% | yo'q | 1* | ha | [−100, 100] |
| NoisyRastrigin | C6 | 1.0e+00 | 88.5% | yo'q | 20 | ha | [−5.12, 5.12] |
| NoisySphere | C7 | 1.0e+00 | **0.0%** | **ha** | 1 | aylantirish ta'sirsiz | [−100, 100] |
| DynamicSphere | C8 | 1.0e+00 | **0.0%** | **ha** | 1 | aylantirish ta'sirsiz | [−100, 100] |

**O'lchash uslubi (10D, rotated trek):**
- `cond(H)` — optimum yaqinidagi son-raqamli Gessianning eigenqiymatlari nisbati.
- `Off-diagonal %` — domen bo'ylab **8 ta tasodifiy nuqtada** Gessianning
  diagonaldan tashqari Frobenius normasi ulushi. 0% = separabel. *(Faqat
  optimum yaqinida o'lchash xato beradi: aylantirilgan Rastrigin u yerda
  sun'iy ravishda separabel ko'rinadi, chunki uning Gessiani optimumda
  birlik matritsaga proporsional.)*
- `Lokal min` — to'liq domen bo'ylab 1-D kesimlarda (6 yo'nalish, 4001 nuqta)
  topilgan maksimal lokal minimumlar soni. Bu **quyi chegara**: 1-D kesim
  ko'p o'lchamli multimodallikni kam baholaydi.
- `*` Composition uchun 1 — kesim artefakti: sphere komponenti domen
  bo'ylab ustun, Ackley to'lqinlari nisbatan ko'rinmay qoladi.

**Separabellik balansi:** 12 tadan **9 tasi separabel emas** (rotated trek).
Asl `shift` trekida esa 12 tadan **faqat 1 tasi** aylantirilgan edi — bu
koordinata bo'yicha ishlaydigan binomial crossover'ni sun'iy ustun qilardi.

---

## 4. Eksperiment dizayni

| Faktor | Darajalar | Soni |
|---|---|---|
| Algoritmlar | TEMOA_V10, TEMOA_V11, jSO, LSHADE, DE, PSO, GWO, WOA, SCA, HHO + 2 nazorat | 12 |
| Funksiyalar | 3-jadvaldagi 12 ta | 12 |
| O'lchamlar | 30, 50, 100 | 3 |
| Yurishlar | mustaqil, urug'lari qayd etilgan | 30 |
| **Asosiy dizayn** | 12 × 12 × 3 × 30 | **12 960 yurish** |

| O'lcham | FES/yurish | Yurishlar | Jami FES |
|---|---|---|---|
| 30D | 90 000 | 4 320 | 389 mln |
| 50D | 150 000 | 4 320 | 648 mln |
| 100D | 300 000 | 4 320 | 1 296 mln |
| **Jami** | | **12 960** | **2.33 mlrd** |

**Qayd etiladigan ko'rsatkichlar:** best / worst / mean / std / **median** / IQR,
error `f(x) − f*`, konvergensiya egri chizig'i (100 nuqta), har yurishning
wall-clock vaqti.

**Statistik tahlil:** har o'lcham uchun alohida Friedman + Iman–Davenport,
Holm post-hoc (nazoratga nisbatan), Nemenyi critical difference, juftlangan
Wilcoxon signed-rank (katak ichida **va** butun oila bo'yicha Holm),
Vargha–Delaney Â₁₂ effekt hajmi.

---

## 5. Ablatsiya va sezgirlik qamrovi

**Ablatsiya — 15 konfiguratsiya** (`experiments/ablation.py`), 8 funksiya, 15 yurish = 1 800 yurish:

| Guruh | Konfiguratsiyalar | Savol |
|---|---|---|
| V10 operator portfeli | full(0,1,2,3), op0, op0+1, op0+2, op0+3, drop-op2 | Qaysi operator hissa qo'shadi? |
| V10 boshqa komponentlar | no-eigen, no-ES-tail, no-CR-floor | Har biri kerakmi? |
| V11 komponentlari | full, no-diversity-guard, no-eigen-gate, op2-qaytarilgan, no-eigen | Tuzatishlar haqiqatan ishlaydimi? |
| Mos yozuvlar | jSO | Gibridsiz yadro |

**Sezgirlik — 6 sweep, 27 qiymat** (`experiments/sensitivity.py`), 4 funksiya, 15 yurish = 1 620 yurish:

| Parametr | Qiymatlar |
|---|---|
| V10.POP_FACTOR | 3, 6, 12, 18, 25 |
| V10.P_MIN | 0.0, 0.02, 0.05, 0.10, 0.20 |
| V10.LS_FRACTION | 0.0, 0.05, 0.10, 0.20 |
| V11.POP_FACTOR | 6, 12, 18, 25 |
| V11.DIV_THRESH | 0.0, 1e-5, 1e-4, 1e-3, 1e-2 |
| V11.DIV_BOOST | 0.3, 0.5, 0.7, 0.9 |

---

## 6. Kategoriya bo'yicha hozirgi natija (30D, 12 yurish)

Har bir algoritmning kategoriya ichidagi **o'rtacha ranki** (1 = eng yaxshi,
12 ta algoritm orasida):

| Kategoriya | V11 | V10 | jSO | LSHADE | DE | PSO | SCA | HHO | GWO | WOA | G'olib |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 unimodal ill-cond. | **2.0** | **2.0** | 3.0 | 3.0 | 5.0 | 6.7 | 7.7 | 6.7 | 9.7 | 11.7 | V11 = V10 |
| C2 tor vodiy | 2.0 | **1.0** | 3.0 | 4.0 | 5.0 | 8.0 | 7.0 | 6.0 | 10.0 | 11.0 | V10 |
| C3 multimodal non-sep. | **2.0** | 2.5 | 3.7 | 4.0 | 2.8 | 7.3 | 6.3 | 8.3 | 9.7 | 12.0 | V11 |
| C4 deceptive multimodal | 3.0 | 5.0 | 2.0 | **1.0** | 4.0 | 6.0 | 9.0 | 8.0 | 12.0 | 10.0 | **L-SHADE** |
| C5 kompozitsiya | 3.0 | **1.0** | 4.0 | 5.0 | 2.0 | 8.0 | 7.0 | 6.0 | 9.0 | 10.0 | V10 |
| C6 multimodal + shovqin | **1.0** | 4.0 | 3.0 | 2.0 | 7.0 | 5.0 | 6.0 | 10.0 | 11.0 | 9.0 | V11 |
| C7 unimodal + shovqin | **1.0** | 3.0 | 2.0 | 4.0 | 5.0 | 8.0 | 7.0 | 6.0 | 9.0 | 11.0 | V11 |
| C8 dinamik | 3.0 | 2.0 | 5.0 | 7.0 | **1.0** | 6.0 | 8.0 | 4.0 | 10.0 | 9.0 | **DE** |

**O'qish qoidalari — bularsiz jadval chalg'itadi:**

1. **C2, C4, C5, C6, C7, C8 da atigi 1 ta funksiya bor.** U yerdagi "g'olib"
   bitta funksiyadagi bitta median — bu kategoriya haqida hech narsa
   isbotlamaydi.
2. **Hech bir DE-oilasi farqi statistik ahamiyatli emas.** Holm post-hoc'da
   V11 ning V10, jSO, L-SHADE va DE dan farqi ahamiyatsiz (p_holm > 0.80).
   Nemenyi CD = 4.81, yuqoridagi rank farqlari esa 1.7 dan kichik.
3. **12 yurishda oila darajasidagi ahamiyatlilikka erishib bo'lmaydi** —
   minimal p = 4.9e-04, Holm 3.8e-04 talab qiladi. 30 yurishda min p = 1.9e-09.
4. **Ishonchli aytilishi mumkin bo'lgan yagona narsa:** butun DE oilasi
   (V11, V10, jSO, L-SHADE, DE) swarm usullaridan (PSO, SCA, HHO, GWO, WOA)
   aniq va statistik jihatdan ahamiyatli darajada ustun. Bu esa yangilik emas.

**Yo'nalish sifatida ko'rinayotgani** (tasdiqlash uchun 30 yurish kerak):
V11 shovqinli (C6, C7) va multimodal non-separabel (C3) masalalarda oldinda;
V10 tor vodiy (C2) va kompozitsiyada (C5); L-SHADE deceptive multimodal (C4)
da aniq ustun; oddiy DE dinamik masalada (C8) hammasini yutadi.

---

## 7. Qamrov TASHQARISIDA — halol bo'shliqlar tahlili

Hozirgi tadqiqot **faqat cheklanmagan uzluksiz bir maqsadli optimizatsiya**
bilan cheklangan. Sizning tadqiqot yo'nalishingizga (matematika, kiberxavfsizlik,
AI) nisbatan quyidagilar **umuman qamralmagan**:

| Soha | Nima yo'q | Nega muhim | Qancha ish |
|---|---|---|---|
| **Matematika** | Cheklovli optimizatsiya (constraint handling) | Muhandislik masalalari deyarli har doim cheklovli; "amaliy ahamiyat" da'vosi shusiz asossiz | Constraint handling + 5–7 ta standart muhandislik masalasi |
| | Kombinator / diskret masalalar | Binar va permutatsion variantlar butunlay boshqa operator talab qiladi | Binar kodlash (transfer funksiya) + TSP/knapsack |
| | Ko'p maqsadli (multi-objective) | Pareto front, IGD/HV metrikalari | Katta ish — alohida maqola |
| | Katta o'lcham (500D, 1000D) | Eigen-crossover `O(D³)` — bu yerda qulaydi (§5) | Cheklangan xotirali eigen variant |
| | **CEC standart to'plamlari** | Hozirgi 12 ta klassik funksiya reviewer uchun yetarli emas; kategoriya darajasidagi da'vo uchun ham kam (§2) | **CEC'2017 (30 funksiya) — eng yuqori ustuvorlik** |
| **Kiberxavfsizlik** | IDS/anomaliya aniqlash uchun feature selection | Binar optimizatsiya + real datasetlar (NSL-KDD, CIC-IDS2017, UNSW-NB15) | Binar variant + 3–4 dataset |
| | Malware/traffic klassifikatsiyasi uchun feature selection | Yuqori o'lchamli, shovqinli, nomutanosib sinflar | Yuqoridagi bilan birga |
| | Aniqlash modellarini sozlash | Hyperparameter optimizatsiya, lekin xavfsizlik metrikalari bilan (FPR cheklovi ostida TPR) | Cheklovli HPO |
| **AI** | Hyperparameter optimizatsiya | Qimmat, shovqinli, aralash (uzluksiz+diskret) qidiruv fazosi | 2–3 model × 3–4 dataset |
| | Neural architecture search | Juda qimmat baholash; surrogate kerak | Katta ish |
| | Tarmoq pruning / siqish | Binar maska optimizatsiyasi | O'rtacha |

**Amaliy tavsiya — ustuvorlik tartibida:**

1. **CEC'2017 (30 funksiya, 10/30/50/100D)** — bu eng katta yutuq. Kategoriya
   darajasidagi da'volarni va Nemenyi CD ni ham hal qiladi (§2, §6).
2. **Cheklovli muhandislik masalalari** (pressure vessel, welded beam, spring,
   speed reducer, 3-bar truss) — amaliy ahamiyat uchun eng arzon yo'l.
3. **Binar variant + IDS feature selection** — sizning kiberxavfsizlik
   yo'nalishingizga to'g'ridan-to'g'ri mos, va gibridning binar fazodagi
   xatti-harakati mustaqil qiziqarli natija.
4. Ko'p maqsadli va NAS — keyingi maqolalar uchun.

---

## 8. Xulosa: hozirgi kuch sinash maydoni

> TEMOA_V11 **11 ta raqobatchi** bilan, **8 ta kategoriyaga** bo'lingan
> **12 ta funksiyada**, **3 ta o'lchamda**, **30 yurishda** — jami **12 960
> yurish** va **2.33 milliard funksiya baholashi** bo'yicha sinaladi.
> Raqobatchilar tasodifiy emas: ularning uchtasi (GWO, WOA, HHO) gibridning
> o'z operatorlari manbasi, ikkitasi (jSO, L-SHADE) esa uning yadrosi.

Bu — **metodologik jihatdan to'g'ri qurilgan, lekin hali tor** maydon. Tor
bo'lgani: 8 kategoriyaning 5 tasida bittadan funksiya, va butun to'plam
cheklanmagan uzluksiz optimizatsiya bilan chegaralangan. §7 dagi birinchi
ikki punkt bu bo'shliqning katta qismini yopadi.
