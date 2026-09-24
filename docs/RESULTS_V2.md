# ACE-SHADE: CEC-2022 natijalari

Avtomatik yaratilgan: 2026-09-24 22:31 UTC. Xom ma'lumot `results/v2/raw_data/`, jadvallar `results/v2/tables/`, grafiklar `results/v2/figures/`.

## 1. Xulosa

**ACE-SHADE eng yaxshi emas.** Eng yaxshi o'rtacha rank: jSO (3.54); ACE-SHADE 4.71.

## 2. Protokol

- Baholash to'plami: **CEC-2022**, 24 instansiya (12 funksiya x 2 o'lcham)
- Byudjet: rasmiy protokol - D=10 -> 200,000, D=20 -> 1,000,000 FES
- Mustaqil runlar: 30
- Seed sxemasi: `42 + 1000*D + run`; berilgan (o'lcham, run) da barcha algoritmlar bir xil boshlang'ich holatdan boshlaydi
- Aniqlik chegarasi: `1e-08` (CEC konventsiyasi); 2136 natija 0 ga tenglashtirildi, xom qiymatlar saqlandi

### 2.1 Sozlash

- Sozlash to'plami: **CEC-2017** (10 funksiya) - CEC-2022 bilan bitta ham umumiy funksiya yo'q
- Byudjet rejimi baholash bilan **aynan bir xil**
- Sozlangan parametrlar: `R_N = 18`, `ETA = 0.1` (rn18_eta0.1)
- Qolgan barcha parametrlar manba usullari e'lon qilgan qiymatda

> **Cheklov.** Tanlangan `R_N` sozlash panjarasining chekkasida. Haqiqiy optimum undan tashqarida bo'lishi mumkin.

## 3. Raqiblar va da'volarning ajratilishi

Raqiblar ikki guruhga bo'lingan va da'volar ham shunga mos ajratilgan.

**Asosiy guruh** (LSHADE-cnEpSin, jSO, L-SHADE, CMA-ES) - oldingi versiyada sinovdan o'tgan amalga oshirishlar. Har biri ACE-SHADE yaxshilashni da'vo qilayotgan aniq narsaning manbai: LSHADE-cnEpSin va CMA-ES kovariatsiya da'vosiga, jSO va L-SHADE rejim ansambli da'vosiga qarshi turadi. **Asosiy da'vo shu guruhga nisbatan qo'yiladi.**

**Ikkilamchi guruh** (NL-SHADE-RSP, LSHADE-SPACMA, LSHADE-RSP) - e'lon qilingan mexanizm ta'rifi bo'yicha qayta yozilgan. Ish muhitida asl maqolalarga va mualliflar kodiga kirish bo'lmagan, shuning uchun ba'zi ikkilamchi tanlovlar (populyatsiya hajmi formulasi, ayrim konstantalar) tasdiqlanmagan. **Bu guruhga nisbatan g'alaba kuchsiz dalil**: agar bizning variantimiz e'lon qilingan natijadan past ishlasa, taqqoslash ularning foydasiga emas.

## 4. Umumiy ranklar va Friedman testi

```
Friedman chi2 = 51.6882,  p = 6.7258e-09
instansiyalar = 24,  algoritmlar = 8
```

| Algoritm | Guruh | O'rtacha rank |
|---|---|---|
| jSO | asosiy | 3.54 |
| LSHADE-RSP | ikkilamchi | 3.65 |
| NL-SHADE-RSP | ikkilamchi | 3.83 |
| L-SHADE | asosiy | 4.19 |
| LSHADE-SPACMA | ikkilamchi | 4.38 |
| LSHADE-cnEpSin | asosiy | 4.58 |
| **ACE-SHADE** | **taklif** | **4.71** |
| CMA-ES | asosiy | 7.12 |

## 5. Post-hoc taqqoslash (nazorat: ACE-SHADE)

| Raqib | Guruh | Rank farqi | z | p (tuzatilmagan) | p (Holm) | Ahamiyatli? |
|---|---|---|---|---|---|---|
| CMA-ES | asosiy | +2.42 | 3.42 | 6.316e-04 | 0.0044 | **ha** |
| jSO | asosiy | -1.17 | -1.65 | 9.896e-02 | 0.5938 | yo`q |
| LSHADE-RSP | ikkilamchi | -1.06 | -1.50 | 1.329e-01 | 0.6647 | yo`q |
| NL-SHADE-RSP | ikkilamchi | -0.87 | -1.24 | 2.159e-01 | 0.8637 | yo`q |
| LSHADE-cnEpSin | asosiy | -0.12 | -0.18 | 8.597e-01 | 1.0000 | yo`q |
| LSHADE-SPACMA | ikkilamchi | -0.33 | -0.47 | 6.374e-01 | 1.0000 | yo`q |
| L-SHADE | asosiy | -0.52 | -0.74 | 4.614e-01 | 1.0000 | yo`q |

## 6. Instansiya bo'yicha g'alaba/teng/mag'lubiyat

Mann-Whitney U + Holm tuzatishi, `alpha = 0.05`.

| Raqib | Guruh | + | = | − |
|---|---|---|---|---|
| NL-SHADE-RSP | ikkilamchi | 4 | 13 | 7 |
| LSHADE-SPACMA | ikkilamchi | 4 | 16 | 4 |
| LSHADE-cnEpSin | asosiy | 2 | 16 | 6 |
| LSHADE-RSP | ikkilamchi | 5 | 11 | 8 |
| jSO | asosiy | 5 | 11 | 8 |
| L-SHADE | asosiy | 4 | 13 | 7 |
| CMA-ES | asosiy | 14 | 7 | 3 |

## 7. Kategoriya bo'yicha tahlil

To'liq jadvallar: `results/v2/tables/category_summary.md`. Kategoriya ranklari: `category_ranks.csv`, o'lcham kesimi: `dimension_category.csv`.

| Kategoriya | n | ACE-SHADE | NL-SHADE-RSP | LSHADE-SPACMA | LSHADE-cnEpSin | LSHADE-RSP | jSO | L-SHADE | CMA-ES |
|---|---|---|---|---|---|---|---|---|---|
| bir ekstremumli | 2 | **4.50** | 4.50 | 4.50 | 4.50 | 4.50 | 4.50 | 4.50 | 4.50 |
| asosiy ko'p ekstremumli | 8 | **3.50** | 5.00 | 3.75 | 4.38 | 4.50 | 4.25 | 4.38 | 6.25 |
| gibrid | 6 | **6.00** | 3.50 | 4.50 | 5.17 | 2.00 | 3.67 | 3.17 | 8.00 |
| kompozitsiya | 8 | **5.00** | 2.75 | 4.88 | 4.38 | 3.81 | 2.50 | 4.69 | 8.00 |

## 8. Ablatsiya

Har bir hissa alohida o'chirilgan holda o'lchandi.

| Variant | O'rtacha rank |
|---|---|
| N1-off | 1.96 |
| cmu-population | 2.29 |
| ACE-SHADE | 2.75 |
| N2-off | 3.00 |

`N1-off` - to'plamli kovariatsiya o'chirilgan (namunaviy kovariatsiyaga qaytish). `N2-off` - rejim ansambli o'chirilgan (faqat jSO banki). `cmu-population` - rank-mu yangilanishi qabul qilingan avlodlar o'rniga butun populyatsiyadan olinadi.

## 9. Cheklovlar

1. Ikkilamchi guruhdagi uchta raqib qayta amalga oshirilgan va asl manbalar bilan tasdiqlanmagan (3-bo'limga qarang).
2. Sozlash 10 ta CEC-2017 funksiyasida, 10 run bilan o'tkazilgan - bu sozlash shovqinini butunlay yo'qotmaydi.
3. Hisoblash murakkabligi (T0/T1/T2) bu ishda o'lchanmagan.
4. Shovqinli va dinamik masalalar CEC-2022 da yo'q, shuning uchun oldingi versiyada aniqlangan shovqin zaifligi bu yerda tekshirilmaydi va ochiq qoladi.

5. Sozlangan `R_N` panjara chekkasida (2.1-bo'limga qarang).
