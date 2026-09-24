# CBA-SHADE: eksperimental natijalar

To'liq tajriba yakunlandi (GitHub Actions run 1, 2026-09-23). Xom
ma'lumotlar `results/full/raw_data/`, jadvallar `results/full/tables/`,
yaqinlashish grafiklari `results/full/figures/` da.

## 1. Eksperimental protokol

### 1.1 Taqqoslanayotgan algoritmlar

Yetti algoritm bir xil sharoitda taqqoslanadi. Barcha bazaviy usullar shu
repozitoriyda qaytadan amalga oshirilgan va bir xil baholash hisoblagichidan
(`Tracker`) foydalanadi, shuning uchun byudjet hisobi ular orasida bir xil.

| Algoritm | Turi | Manba |
|---|---|---|
| **CBA-SHADE** | taklif etilayotgan | shu ish |
| L-SHADE | DE, chiziqli populyatsiya kamayishi | Tanabe & Fukunaga, 2014 |
| jSO | L-SHADE ning parametr cheklovli varianti | Brest va b., 2017 |
| LSHADE-cnEpSin | L-SHADE + eigen crossover + sinusoidal moslashuv | Awad va b., 2017 |
| SHADE | tarixga asoslangan moslashuvchi DE | Tanabe & Fukunaga, 2013 |
| CMA-ES | kovariatsiya matritsasi moslashuvi | Hansen & Ostermeier, 2001 |
| DE/rand/1/bin | klassik nazorat | Storn & Price, 1997 |

### 1.2 Test to'plami

12 ta funksiya, har biri uchta o'lchamda (30, 50, 100) — jami **36 instansiya**.
Har bir funksiya optimumi tasodifiy siljitilgan, shuning uchun koordinata
boshidagi yechim afzallik bermaydi.

| Funksiya | Xususiyati | Sinov maqsadi |
|---|---|---|
| Ackley, Griewank, Levy, Schwefel | ko'p ekstremumli | global qidiruv |
| Rastrigin (shovqinli) | ko'p ekstremumli + shovqin | shovqinga chidamlilik |
| Sphere (shovqinli), Sphere (dinamik) | bir ekstremumli, o'zgaruvchan | kuzatuv qobiliyati |
| Rosenbrock | tor vodiy | mahalliy model sifati |
| BentCigar, Zakharov, RotatedElliptic | yomon shartlangan | kovariatsiyani o'rganish |
| Composition | aralash | umumiy mustahkamlik |

### 1.3 Byudjet va takrorlashlar

- Baholashlar soni: `FES = 3000 · D` (30D → 90 000; 50D → 150 000; 100D → 300 000)
- Mustaqil ishga tushirishlar: har bir (algoritm, funksiya, o'lcham) uchun **30**
- Jami: 7 × 12 × 3 × 30 = **7 560 ishga tushirish**

### 1.4 Seed sxemasi

Seed `42 + 1000·D + run` formulasi bilan beriladi va har bir ishga tushirishdan
oldin qayta o'rnatiladi. Shuning uchun **berilgan (o'lcham, run) juftligida
barcha algoritmlar bir xil boshlang'ich populyatsiyadan va bir xil siljitish
vektoridan boshlaydi** — taqqoslash juftlashtirilgan bo'ladi va tasodifiy
boshlang'ich holat farqi natijaga ta'sir qilmaydi.

### 1.5 Aniqlik chegarasi

CEC musobaqalari konventsiyasiga muvofiq `1e-8` dan kichik xato nol deb
qabul qilinadi (`PRECISION_FLOOR`). Bu metodologik jihatdan zarur: busiz
`1.5e-32` va `4.1e-31` kabi ikki aniq yechim "g'alaba" va "mag'lubiyat"
sifatida sanaladi va ranklar mashina aniqligidagi shovqin bilan buziladi.
Xom qiymatlar `full_raw_results.csv` da o'zgarishsiz saqlanadi.

### 1.6 Statistik tahlil

1. **Friedman testi** — barcha algoritmlar bo'yicha umumiy farqni tekshiradi;
   χ² statistikasi va Iman–Davenport F tuzatishi bilan.
2. **Post-hoc taqqoslash** — CBA-SHADE nazorat sifatida, har bir raqib bilan
   juftlik taqqoslash; `z` statistikasi, **Holm tuzatishi** bilan `p`
   qiymatlari va Nemenyi tanqidiy farqi (CD).
3. **Mann–Whitney U testi** — har bir instansiyada juftlik taqqoslash,
   Holm tuzatishi bilan; natija `+/=/−` (g'alaba/teng/mag'lubiyat) jadvalida.
4. **Hisoblash murakkabligi** — CEC protokoli bo'yicha `T0`, `T1`, `T2`.

Barcha testlar `α = 0.05` darajasida.

---

## 2. Umumiy natijalar

Quyidagi jadvalda har bir instansiya uchun 30 ta mustaqil ishga tushirishning
o'rtacha xatosi va qavs ichida standart og'ishi keltirilgan. Har bir satrda eng
yaxshi natija qalin ko'rsatilgan. Qiymatlar `1e-8` aniqlik chegarasi
qo'llanganidan keyin hisoblangan; `0` ushbu chegaradan past, ya'ni amalda hal
qilingan degani.

| Funksiya | D | CBA-SHADE | L-SHADE | jSO | LSHADE-cnEpSin | SHADE | CMA-ES | DE |
|---|---|---|---|---|---|---|---|---|
| Ackley | 30 | **0** | **0** | **0** | **0** | **0** | 5.08e+00 (4e+00) | **0** |
| Ackley | 50 | **0** | **0** | **0** | **0** | **0** | 5.60e+00 (5e+00) | 4.70e-01 (7e-1) |
| Ackley | 100 | **0** | 8.53e-09 (8e-9) | **0** | 1.18e-07 (4e-8) | 3.74e-01 (5e-1) | 7.77e+00 (2e+00) | 3.12e+00 (1e+00) |
| BentCigar | 30 | **0** | **0** | **0** | **0** | **0** | 8.80e+08 (1e+09) | **0** |
| BentCigar | 50 | **0** | **0** | **0** | **0** | **0** | 1.73e+09 (1e+09) | 5.03e-03 (3e-2) |
| BentCigar | 100 | **0** | 5.82e-05 (5e-5) | 2.97e-08 (5e-8) | 5.29e-04 (5e-4) | **0** | 3.29e+09 (2e+09) | 2.33e+02 (1e+03) |
| Composition | 30 | **0** | **0** | **0** | **0** | **0** | 5.26e+02 (8e+02) | **0** |
| Composition | 50 | **0** | **0** | **0** | **0** | **0** | 6.55e+02 (7e+02) | 9.35e-10 (5e-9) |
| Composition | 100 | **0** | 1.31e-08 (7e-9) | **0** | 1.92e-07 (9e-8) | **0** | 1.89e+03 (2e+03) | 2.97e-05 (2e-4) |
| DynamicSphere | 30 | **1.47e-02 (6e-3)** | 1.17e+00 (3e-1) | 1.25e+00 (3e-1) | 8.28e-01 (2e-1) | 1.93e+00 (4e-1) | 1.17e+03 (1e+03) | 4.15e-01 (2e-1) |
| DynamicSphere | 50 | **2.75e-02 (7e-3)** | 3.66e+00 (7e-1) | 2.99e+00 (9e-1) | 3.06e+00 (5e-1) | 1.92e+00 (6e-1) | 1.45e+03 (2e+03) | 2.72e+00 (2e+00) |
| DynamicSphere | 100 | **3.00e-01 (5e-2)** | 1.12e+01 (2e+00) | 2.11e+01 (4e+00) | 1.24e+01 (2e+00) | 3.01e+00 (1e+00) | 3.99e+03 (3e+03) | 4.93e+01 (6e+01) |
| Griewank | 30 | **0** | **0** | **0** | **0** | **0** | 7.63e+00 (1e+01) | 2.22e-03 (5e-3) |
| Griewank | 50 | **0** | **0** | **0** | **0** | 2.47e-04 (1e-3) | 1.43e+01 (2e+01) | 2.61e-02 (6e-2) |
| Griewank | 100 | 1.23e-03 (3e-3) | **0** | 2.47e-04 (1e-3) | **0** | 3.61e-03 (7e-3) | 3.40e+01 (3e+01) | 1.55e-01 (4e-1) |
| Levy | 30 | **0** | **0** | **0** | **0** | **0** | 1.31e+01 (1e+01) | 8.88e-01 (2e+00) |
| Levy | 50 | **0** | **0** | **0** | **0** | 7.53e-02 (1e-1) | 3.73e+01 (2e+01) | 1.96e+00 (2e+00) |
| Levy | 100 | 1.05e-01 (1e-1) | **0** | 5.97e-03 (2e-2) | **0** | 2.54e+00 (2e+00) | 9.52e+01 (3e+01) | 1.98e+01 (9e+00) |
| NoisyRastrigin | 30 | **2.01e-01 (4e-2)** | 1.65e+00 (5e-1) | 4.74e+00 (2e+00) | 4.27e+00 (2e+00) | 1.27e+01 (2e+00) | 7.84e+01 (1e+01) | 6.04e+01 (3e+01) |
| NoisyRastrigin | 50 | **6.17e-01 (4e-1)** | 1.31e+01 (2e+00) | 2.04e+01 (3e+00) | 2.31e+01 (4e+00) | 1.21e+01 (3e+00) | 1.40e+02 (3e+01) | 3.93e+01 (2e+01) |
| NoisyRastrigin | 100 | 1.35e+01 (1e+00) | 9.78e+01 (8e+00) | 9.06e+01 (7e+00) | 1.32e+02 (1e+01) | **4.36e+00 (4e-1)** | 3.47e+02 (5e+01) | 1.14e+02 (2e+01) |
| NoisySphere | 30 | 1.36e-02 (4e-3) | 1.27e-02 (3e-3) | **9.02e-03 (3e-3)** | 9.60e-03 (2e-3) | 1.15e-02 (2e-3) | 9.41e+02 (1e+03) | 1.48e-02 (3e-3) |
| NoisySphere | 50 | 2.61e-02 (5e-3) | 1.61e-02 (3e-3) | 1.77e-02 (4e-3) | 1.70e-02 (4e-3) | **1.47e-02 (4e-3)** | 1.77e+03 (2e+03) | 3.28e-02 (1e-2) |
| NoisySphere | 100 | 7.37e-02 (9e-3) | **3.39e-02 (5e-3)** | 4.68e-02 (8e-3) | 5.58e-02 (9e-3) | 5.14e-02 (8e-3) | 3.08e+03 (3e+03) | 1.90e-01 (2e-1) |
| Rosenbrock | 30 | **4.74e-01 (1e+00)** | 1.32e+01 (8e-1) | 1.22e+01 (9e-1) | 1.30e+01 (9e-1) | 7.99e-01 (6e-1) | 3.42e+05 (6e+05) | 3.89e+01 (3e+01) |
| Rosenbrock | 50 | 2.09e+01 (2e+00) | 3.50e+01 (1e+00) | 3.31e+01 (9e-1) | 3.60e+01 (8e-1) | **1.35e+01 (1e+01)** | 1.66e+06 (3e+06) | 3.90e+02 (1e+03) |
| Rosenbrock | 100 | 7.70e+01 (1e+01) | 1.08e+02 (3e+01) | 8.86e+01 (1e+01) | 9.43e+01 (2e+01) | **6.65e+01 (3e+01)** | 2.95e+06 (3e+06) | 2.32e+05 (1e+06) |
| RotatedElliptic | 30 | 3.63e+03 (5e+03) | 2.06e+03 (3e+03) | 1.55e+03 (2e+03) | **3.83e+00 (1e+01)** | 1.14e+05 (7e+04) | 2.83e+06 (5e+06) | 3.79e+06 (1e+06) |
| RotatedElliptic | 50 | 4.80e+04 (2e+04) | 7.23e+04 (4e+04) | 6.57e+04 (2e+04) | **1.34e+04 (1e+04)** | 2.40e+05 (1e+05) | 4.55e+06 (5e+06) | 8.98e+06 (6e+06) |
| RotatedElliptic | 100 | **4.64e+05 (1e+05)** | 9.94e+05 (2e+05) | 1.16e+06 (3e+05) | 4.97e+05 (1e+05) | 9.72e+05 (3e+05) | 8.86e+06 (7e+06) | 1.07e+07 (3e+06) |
| Schwefel | 30 | **3.82e-04** | 1.45e-01 (3e-1) | 3.41e+00 (4e+00) | 1.67e+01 (3e+01) | 2.71e+01 (5e+01) | 5.41e+03 (6e+02) | 2.06e+03 (7e+02) |
| Schwefel | 50 | **6.36e-04 (7e-13)** | 1.09e+02 (7e+01) | 4.87e+02 (2e+02) | 1.17e+03 (2e+02) | 4.78e+01 (7e+01) | 8.99e+03 (9e+02) | 4.39e+03 (8e+02) |
| Schwefel | 100 | 7.24e+01 (5e+01) | 5.27e+03 (4e+02) | 5.85e+03 (6e+02) | 1.01e+04 (8e+02) | **4.78e+01 (9e+01)** | 1.80e+04 (9e+02) | 1.07e+04 (1e+03) |
| Zakharov | 30 | **0** | **0** | **0** | **0** | **0** | 2.03e-01 (1e+00) | 1.37e-01 (2e-1) |
| Zakharov | 50 | **0** | 1.32e-04 (4e-4) | 4.03e-05 (4e-5) | 5.24e-07 (5e-7) | 1.54e-06 (3e-6) | **0** | 8.47e+01 (4e+01) |
| Zakharov | 100 | 5.01e-02 (4e-2) | 6.53e-01 (4e-1) | 2.54e-01 (1e-1) | 3.68e-02 (2e-2) | 2.64e-02 (2e-2) | **0** | 1.13e+03 (2e+02) |

## 3. Friedman testi va ranklar

Friedman testi yetti algoritm orasida sezilarli farq borligini tasdiqlaydi:

```
chi2 = 122.90,  p = 4.00e-24  (36 instansiya, 7 algoritm)
```

| Algoritm | O'rtacha rank |
|---|---|
| **CBA-SHADE** | **2.50** |
| SHADE | 3.14 |
| jSO | 3.28 |
| L-SHADE | 3.38 |
| LSHADE-cnEpSin | 3.40 |
| DE | 5.71 |
| CMA-ES | 6.60 |

CBA-SHADE eng yaxshi o'rtacha rankka ega. Biroq bu ko'rsatkichning o'zi
ustunlikni isbotlamaydi — buning uchun post-hoc taqqoslash zarur.

## 4. Post-hoc taqqoslash (nazorat: CBA-SHADE)

Bu bo'lim ishning eng muhim va ayni paytda eng cheklovchi qismi.

| Raqib | Rank farqi | z | p (tuzatilmagan) | p (Holm) | Ahamiyatli? |
|---|---|---|---|---|---|
| SHADE | 0.64 | 1.25 | 0.210 | 0.305 | **yo'q** |
| jSO | 0.78 | 1.53 | 0.127 | 0.305 | **yo'q** |
| L-SHADE | 0.88 | 1.72 | 0.086 | 0.305 | **yo'q** |
| LSHADE-cnEpSin | 0.90 | 1.77 | 0.076 | 0.305 | **yo'q** |
| DE | 3.21 | 6.30 | 3.0e-10 | 1.5e-09 | ha |
| CMA-ES | 4.10 | 8.05 | 8.9e-16 | 5.3e-15 | ha |

**Asosiy xulosa: CBA-SHADE zamonaviy DE variantlarining hech biridan
statistik jihatdan ustun emas.** Ahamiyatli farq faqat klassik DE va
CMA-ES ga nisbatan kuzatiladi — ya'ni eng kuchsiz ikki bazaviy usulga
nisbatan.

Bu natijani yumshatib ko'rsatish mumkin emas. `p = 0.076` (LSHADE-cnEpSin)
chegaraga yaqin bo'lsa-da, Holm tuzatishidan keyin `0.305` ga ko'tariladi va
`alpha = 0.05` dan ancha uzoq. 36 instansiya va 6 ta taqqoslash uchun
Friedman post-hoc testining quvvati cheklangan; shu sababli ushbu natija
"ustunlik yo'q" emas, "ustunlik isbotlanmadi" deb o'qilishi kerak. Ikkalasi
ham maqola uchun bir xil xulosaga olib keladi: **joriy tajriba CBA-SHADE ning
zamonaviy raqiblardan afzalligini qo'llab-quvvatlamaydi.**

## 5. Instansiya bo'yicha g'alaba/teng/mag'lubiyat

Mann–Whitney U testi (Holm tuzatishi bilan, `alpha = 0.05`) har bir
instansiyada juftlik taqqoslash beradi:

| Raqib | + (yutdi) | = (teng) | − (yutqazdi) |
|---|---|---|---|
| L-SHADE | 19 | 14 | 3 |
| jSO | 17 | 15 | 4 |
| LSHADE-cnEpSin | 16 | 14 | 6 |
| SHADE | 13 | 15 | 8 |
| DE | 24 | 10 | 2 |
| CMA-ES | 31 | 3 | 2 |

Bu jadval post-hoc natijasidan ancha ijobiy ko'rinadi va bu qarama-qarshilik
tushuntirishni talab qiladi. Ikki test turli savollarga javob beradi:

- **Juftlik testi** har bir instansiyada faqat ikki algoritmni taqqoslaydi va
  36 ta mustaqil qaror beradi. Bu yerda CBA-SHADE ko'pincha ustun.
- **Friedman post-hoc** esa butun to'plam bo'yicha ranklarning o'rtacha
  farqiga tayanadi va 6 ta bir vaqtda taqqoslash uchun tuzatiladi. Rank
  farqlari kichik (2.50 va 3.14 oralig'ida), shuning uchun ahamiyat
  darajasiga yetmaydi.

Metodologik jihatdan post-hoc natijasi ustunroq: u ko'p taqqoslashdan kelib
chiqadigan xato ehtimolini nazorat qiladi. Juftlik jadvali qo'shimcha
ma'lumot sifatida keltiriladi, asosiy da'vo sifatida emas.

## 6. O'lcham bo'yicha tahlil

| Algoritm | 30D | 50D | 100D |
|---|---|---|---|
| **CBA-SHADE** | **2.71** | **2.29** | **2.50** |
| SHADE | 3.79 | 3.00 | 2.63 |
| jSO | 3.13 | 3.42 | 3.29 |
| L-SHADE | 3.29 | 3.50 | 3.33 |
| LSHADE-cnEpSin | 3.04 | 3.42 | 3.75 |
| DE | 5.13 | 5.92 | 6.08 |
| CMA-ES | 6.92 | 6.46 | 6.42 |

CBA-SHADE uchala o'lchamda ham eng yaxshi o'rtacha rankka ega. Biroq ikkita
tendentsiya diqqatga sazovor:

1. **SHADE o'lcham ortishi bilan yaqinlashadi.** 30D da farq 1.08 rank,
   100D da atigi 0.13. Ya'ni eng sodda zamonaviy bazaviy usul yuqori
   o'lchamlarda deyarli teng natija beradi.
2. **LSHADE-cnEpSin teskari yo'nalishda harakat qiladi** (3.04 → 3.75), lekin
   quyida ko'rsatilganidek, aynan u `RotatedElliptic` da CBA-SHADE dan keskin
   ustun.

## 7. Hisoblash murakkabligi

CEC protokoli bo'yicha 10D da o'lchangan:

| T0 | T1 | T2 (o'rtacha) | (T2−T1)/T0 |
|---|---|---|---|
| 0.253 s | 1.716 s | 4.930 s | **12.71** |

Nisbat 12.71 — CBA-SHADE bir xil byudjet uchun sof funksiya baholashdan
taxminan o'n uch barobar ko'proq vaqt talab qiladi. Bu kovariatsiya
matritsasining davriy eigen-dekompozitsiyasi bilan izohlanadi.

**Cheklov:** murakkablik faqat 10D da o'lchangan. CEC protokoli uni har bir
o'lcham uchun talab qiladi va eigen-dekompozitsiya `O(D^3)` bo'lgani uchun
100D da nisbat sezilarli darajada yomonlashishi kutiladi. Bu maqola uchun
to'ldirilishi shart.

## 8. Zaif tomonlar va cheklovlar

Bu bo'lim har bir instansiyada CBA-SHADE ni **eng yaxshi raqib** bilan
taqqoslaydi — 5-bo'limdagi juftlik taqqoslashdan ancha qattiqroq mezon.

CBA-SHADE 36 instansiyadan **8 tasida eng yaxshi**, 18 tasida statistik jihatdan teng, **10 tasida yutqazdi**.

### Statistik jihatdan ahamiyatli mag'lubiyatlar

| Funksiya | D | CBA-SHADE | Yutgan algoritm | Uning natijasi | Nisbat | p (Holm) |
|---|---|---|---|---|---|---|
| Levy | 100 | 1.047e-01 | L-SHADE | 0.000e+00 | infx | 0.000 |
| Zakharov | 100 | 5.014e-02 | CMA-ES | 0.000e+00 | infx | 0.000 |
| RotatedElliptic | 30 | 3.634e+03 | LSHADE-cnEpSin | 3.831e+00 | 948.7x | 0.000 |
| RotatedElliptic | 50 | 4.803e+04 | LSHADE-cnEpSin | 1.337e+04 | 3.6x | 0.000 |
| NoisyRastrigin | 100 | 1.352e+01 | SHADE | 4.360e+00 | 3.1x | 0.000 |
| NoisySphere | 100 | 7.373e-02 | L-SHADE | 3.391e-02 | 2.2x | 0.000 |
| NoisySphere | 50 | 2.609e-02 | SHADE | 1.474e-02 | 1.8x | 0.000 |
| Rosenbrock | 50 | 2.086e+01 | SHADE | 1.355e+01 | 1.5x | 0.000 |
| Schwefel | 100 | 7.239e+01 | SHADE | 4.781e+01 | 1.5x | 0.000 |
| NoisySphere | 30 | 1.362e-02 | jSO | 9.024e-03 | 1.5x | 0.000 |

### Teng chiqqan instansiyalar

Bu instansiyalarda farq Holm tuzatishidan keyin ahamiyatli emas (`p >= 0.05`), ya'ni ustunlik da'vo qilib bo'lmaydi:

| Funksiya | D | CBA-SHADE | Eng yaqin raqib | Uning natijasi | p (Holm) |
|---|---|---|---|---|---|
| Ackley | 30 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| Ackley | 50 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| Ackley | 100 | 0.000e+00 | jSO | 0.000e+00 | 1.000 |
| BentCigar | 30 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| BentCigar | 50 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| BentCigar | 100 | 0.000e+00 | SHADE | 0.000e+00 | 1.000 |
| Composition | 30 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| Composition | 50 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| Composition | 100 | 0.000e+00 | jSO | 0.000e+00 | 1.000 |
| Griewank | 30 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| Griewank | 50 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| Griewank | 100 | 1.232e-03 | L-SHADE | 0.000e+00 | 0.168 |
| Levy | 30 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| Levy | 50 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| Rosenbrock | 100 | 7.701e+01 | SHADE | 6.647e+01 | 0.054 |
| RotatedElliptic | 100 | 4.643e+05 | LSHADE-cnEpSin | 4.971e+05 | 0.326 |
| Zakharov | 30 | 0.000e+00 | L-SHADE | 0.000e+00 | 1.000 |
| Zakharov | 50 | 0.000e+00 | CMA-ES | 0.000e+00 | 1.000 |

### 8.1 Tizimli zaifliklar

Mag'lubiyatlar tasodifiy emas, uchta aniq guruhga bo'linadi:

**(a) Shovqin ostidagi BIR EKSTREMUMLI masalalar.** Dastlab bu zaiflik
"shovqinli funksiyalar" deb umumlashtirilgan edi; batafsil tekshiruv bu
tavsifni rad etadi. Haqiqiy manzara ancha aniqroq:

| Funksiya | 30D | 50D | 100D |
|---|---|---|---|
| `NoisySphere` (bir ekstremumli) | yutqazdi 1.5x | yutqazdi 1.8x | yutqazdi 2.2x |
| `NoisyRastrigin` (ko'p ekstremumli) | **yutdi 8.2x** | **yutdi 19.5x** | yutqazdi 3.1x |

Ya'ni muammo shovqinning o'zida emas. Ko'p ekstremumli shovqinli masalada
CBA-SHADE raqiblardan bir necha barobar ustun — populyatsiyaga asoslangan
qidiruv shovqinni tabiiy ravishda o'rtachalaydi. Bir ekstremumli shovqinli
masalada esa aksincha: CMA quyrug'i shovqindan kovariatsiya modelini
quradi va shovqinni signal deb qabul qiladi, holbuki bu yerda modelning
foydasi eng kam (Sphere uchun kovariatsiyani o'rganish keraksiz).

Shu sababli tuzatish yo'nalishi ham torroq va aniqroq: quyruqni butunlay
qayta ishlash emas, balki shovqin aniqlanganda va masala bir ekstremumli
ko'ringanda uni o'chirish yoki baholashlarni qayta o'rtachalash kifoya.

Sozlash bosqichida quyruq byudjetini 5% dan 15% ga oshirish `Rosenbrock` va
`DynamicSphere` da katta yutuq bergan edi; ayni o'zgarish `NoisySphere` da
zarar keltirgan ko'rinadi, lekin bu almashuv sozlash paytida o'lchanmagan.

**(b) `RotatedElliptic` — ma'lum va tasdiqlangan zaiflik.** 30D da 949x,
50D da 3.6x farq bilan LSHADE-cnEpSin dan yutqazadi (100D da teng chiqadi).
Sabab: cnEpSin eigen-crossover ni butun populyatsiyaga bir vaqtda qo'llaydi
va `18*D` populyatsiya bilan ishlaydi, CBA-SHADE esa kovariatsiyani faqat
quyruq bosqichida to'liq ishlatadi.

**(c) 100D da aniqlikka yetib bormaslik.** `Levy` va `Zakharov` da raqib
aniq yechimni topadi (`0`), CBA-SHADE esa `1.0e-01` va `5.0e-02` da
to'xtaydi. Bu yuqori o'lchamda yaqinlashish tezligining yetishmasligini
ko'rsatadi.

### 8.2 G'alaba va mag'lubiyatlarning kattaligi

Mag'lubiyatlar sonini sanash yetarli emas — ularning kattaligi ham muhim.
Taqsimot sezilarli darajada nosimmetrik:

| Eng katta g'alabalar | | Eng katta mag'lubiyatlar | |
|---|---|---|---|
| `Schwefel` 50D | **75 060x** | `RotatedElliptic` 30D | 949x |
| `Schwefel` 30D | **380x** | `RotatedElliptic` 50D | 3.6x |
| `DynamicSphere` 50D | **70x** | `NoisyRastrigin` 100D | 3.1x |
| `NoisyRastrigin` 50D | **19.5x** | `NoisySphere` 100D | 2.2x |
| `DynamicSphere` 100D | **10x** | `NoisySphere` 50D | 1.8x |

`RotatedElliptic` dan tashqari barcha mag'lubiyatlar 1.5–3.1 barobar
oralig'ida, g'alabalar esa 1.7 barobardan 75 000 barobargacha. Boshqacha
aytganda, usul yutqazganda kam yutqazadi, yutganda ko'p yutadi. Bu
o'rtacha rank ko'rsatkichida ko'rinmaydi, chunki rank kattalikni hisobga
olmaydi.

### 8.3 Ahamiyatlilikka qancha qolgan?

Quyidagi hisob 8.1 da aniqlangan oltita zaif instansiya (`NoisySphere`
uchala o'lchamda, `NoisyRastrigin` 100D, `RotatedElliptic` 30D va 50D)
eng yaxshi raqib darajasiga **tenglashtirilsa** post-hoc natijasi qanday
bo'lishini ko'rsatadi. Bu haqiqiy natija emas, balki mavjud imkoniyatni
baholash uchun sezgirlik tahlili:

| Raqib | Joriy Holm p | Faraziy Holm p |
|---|---|---|
| LSHADE-cnEpSin | 0.305 | **0.024** |
| L-SHADE | 0.305 | **0.024** |
| jSO | 0.305 | **0.026** |
| SHADE | 0.305 | **0.029** |

Oltita instansiyani tenglashtirish barcha to'rtta taqqoslashni
ahamiyatlilik chegarasidan o'tkazadi. Demak joriy natijaning
ahamiyatsizligi umumiy zaiflikdan emas, aniq va lokal muammolardan kelib
chiqadi. Bu tuzatishlar amalga oshsa, ustunlik da'vosi statistik
jihatdan asoslangan bo'lishi mumkin — lekin bu **kafolat emas**, chunki
hisob tenglashtirish muvaffaqiyatli bo'lishini faraz qiladi.

### 8.4 SHADE bilan taqqoslash alohida e'tiborni talab qiladi

SHADE — taqqoslanayotgan zamonaviy usullarning eng soddasi — umumiy rankda
ikkinchi o'rinda (3.14) va CBA-SHADE ni **4 ta instansiyada** yengadi
(`NoisyRastrigin` 100D, `NoisySphere` 50D, `Rosenbrock` 50D, `Schwefel`
100D). 100D da ranklari deyarli teng (2.50 va 2.63).

Bu CBA-SHADE ning qo'shimcha mexanizmlari (kovariatsiyani o'rganish,
parametr rejimi ansambli, CMA quyrug'i) joriy byudjetda o'zini
oqlayotganiga shubha tug'diradi. Maqolada bu savol ochiq qo'yilishi va
ablatsiya natijalari (`docs/ABLATION.md`) bilan birga muhokama qilinishi
kerak.

### 8.5 Eksperimental cheklovlar

1. **Test to'plami standart emas.** 12 funksiya qo'lda tanlangan; CEC-2017
   yoki CEC-2022 kabi tan olingan to'plam ishlatilmagan, shuning uchun
   natijalarni adabiyotdagi qiymatlar bilan bevosita taqqoslab bo'lmaydi.
2. **Byudjet kichik.** `3000*D` — CEC protokolidagi `10000*D` dan uch
   barobar kam. L-SHADE va jSO aynan kattaroq byudjet uchun sozlangan,
   shuning uchun ular bu yerda o'z imkoniyatini to'liq ko'rsatmayapti.
   Bu taqqoslashni CBA-SHADE foydasiga qiya qiladi.
3. **30 ta ishga tushirish.** CEC standarti 51 ni talab qiladi; kamroq run
   statistik quvvatni pasaytiradi.
4. **Parametrlar shu to'plamda sozlangan.** 30D dagi o'sha 12 funksiya ham
   sozlash, ham baholash uchun ishlatilgan. Bu optimistik qiyalik
   kiritadi — mustaqil validatsiya to'plami yo'q.
5. **Murakkablik faqat 10D da o'lchangan** (7-bo'limga qarang).

## 9. Maqola uchun keyingi qadamlar

Joriy natijalar Q1 darajasidagi jurnal uchun yetarli emas, chunki asosiy
da'vo — zamonaviy usullardan ustunlik — statistik jihatdan
qo'llab-quvvatlanmagan. Quyidagi qadamlar zarur:

1. **CEC-2017 yoki CEC-2022 to'plamiga o'tish**, `10000*D` byudjet va 51 run
   bilan. Bu uch cheklovni bir vaqtda hal qiladi va natijalarni adabiyot
   bilan taqqoslanadigan qiladi.
2. **Bir ekstremumli shovqinli masalalarda quyruqni o'chirish.** 8.1(a)
   ko'rsatganidek, muammo shovqinda emas, shovqin ostida kovariatsiya
   modelini qurishda. Eng arzon yechim: quyruq bosqichiga o'tishdan oldin
   eng yaxshi nuqtani ikki marta baholab shovqin darajasini o'lchash va
   shovqin sezilarli bo'lsa quyruqni o'tkazib yuborish. Bu 8.4-bo'limdagi
   hisobga ko'ra eng yuqori foyda beradigan o'zgarish.
3. **`RotatedElliptic` uchun kovariatsiyani asosiy bosqichga chiqarish**,
   cnEpSin uslubida. Sozlash ko'rsatdiki, populyatsiyani `8*D` ga oshirish
   bu funksiyada 2686 → 305 yaxshilanish beradi, lekin boshqa joyda
   zarar keltiradi; adaptiv yechim kerak.
4. **Mustaqil validatsiya to'plami** — sozlash va baholash funksiyalarini
   ajratish.
5. **Zamonaviyroq raqiblar qo'shish**: AGSK, NL-SHADE-RSP, EA4eig — joriy
   bazaviy usullar 2013–2017 yillarga tegishli.
6. **Murakkablikni uchala o'lchamda o'lchash.**

Agar bu qadamlardan keyin ham ustunlik isbotlanmasa, maqolani boshqa
yo'nalishda — masalan, komponentlar tahlili yoki moslashuv mexanizmi
bo'yicha metodologik ish sifatida — shakllantirish maqsadga muvofiq.
