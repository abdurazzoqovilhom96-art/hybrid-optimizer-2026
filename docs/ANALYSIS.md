# Oxirgi eksperiment natijalarining tahlili (TEMOA_V10_HYBRID)

Tahlil qilingan fayllar: `full_raw_results.csv` (7560 qator), `summary_mean_std.csv`,
`summary_results.csv`, `overall_average_ranks.csv`, `friedman_test.txt`.

**Protokol:** 7 algoritm x 12 funksiya x 3 o'lcham (30/50/100) x 30 mustaqil run;
byudjet `FES = 3000 x D`; shovqinli masalalarda yakuniy qiymat shovqinsiz
`true_func` orqali qayta hisoblangan.

---

## 1. Umumiy manzara

Friedman testi: `chi2 = 185.33`, `p = 2.50e-37` (N = 36 instansiya, k = 7 algoritm).
Iman-Davenport F = 211.52. Ya'ni "algoritmlar orasida farq yo'q" gipotezasi
qat'iy rad etiladi.

| Algoritm | O'rtacha rank (mean jadval) | O'rtacha rank (mediana jadval) |
|---|---|---|
| TEMOA_V10_HYBRID | **1.139** | **1.181** |
| DE | 1.861 | 1.819 |
| SCA | 3.083 | 3.083 |
| GWO | 4.556 | 4.528 |
| PSO | 5.306 | 5.361 |
| WOA | 5.722 | 5.694 |
| HHO | 6.333 | 6.333 |

Birinchi qarashda natija a'lo ko'rinadi. Ikkita tekshiruv esa buni buzadi.

---

## 2. Birinchi muammo: post-hoc test ustunlikni tasdiqlamaydi

Friedman testining o'zi faqat "kamida bitta algoritm boshqasidan farq qiladi"
deydi. Ustunlikni *post-hoc* test isbotlaydi. Nazorat algoritmi sifatida
TEMOA olinganda (Bonferroni–Dunn / Holm, `SE = 0.5092`):

| Raqib | R_i | R_i − R_nazorat | z | p (Holm) | Sezilarli? |
|---|---|---|---|---|---|
| HHO | 6.333 | 5.194 | 10.202 | < 1e-15 | ha |
| WOA | 5.722 | 4.583 | 9.001 | < 1e-15 | ha |
| PSO | 5.306 | 4.167 | 8.183 | 8.9e-16 | ha |
| GWO | 4.556 | 3.417 | 6.710 | 5.8e-11 | ha |
| SCA | 3.083 | 1.944 | 3.819 | 2.7e-04 | ha |
| **DE** | **1.861** | **0.722** | **1.418** | **0.156** | **YO'Q** |

Kritik farq: `CD(Nemenyi, 0.05) = 1.502`, `CD(Bonferroni–Dunn) = 1.343`.
Kuzatilgan farq **0.722** — ikkala chegaradan ham kichik.

> **Xulosa:** rank tahlili bo'yicha TEMOA_V10_HYBRID oddiy `DE/rand/1/bin` dan
> statistik jihatdan ustun emas. Q1 darajadagi jurnal retsenzenti birinchi
> navbatda shu hisob-kitobni qiladi.

Instansiya bo'yicha Mann–Whitney U + Holm (alfa = 0.05) natijasi:

| Raqib | + (yutdi) | = (teng) | − (yutqazdi) |
|---|---|---|---|
| DE | 26 | 4 | **6** |
| GWO | 36 | 0 | 0 |
| WOA | 36 | 0 | 0 |
| PSO | 36 | 0 | 0 |
| SCA | 36 | 0 | 0 |
| HHO | 36 | 0 | 0 |

36/0/0 ko'rinishidagi to'liq g'alaba faqat metafora asosidagi algoritmlarga
qarshi olingan. Yagona jiddiy raqib — DE.

---

## 3. Ikkinchi muammo: raqiblar to'plami zaif

Bu eng og'ir masala. TEMOA_V10_HYBRID ning yadrosi — L-SHADE / jSO
(current-to-pbest-w/1 + arxiv + success-history + LPSR), eigen-crossover esa
LSHADE-cnEpSin dan olingan. Ya'ni algoritm o'z **ota-onalari** bilan umuman
taqqoslanmagan.

Shu ota-onalarni o'sha protokolda (30D, `FES = 90000`, bir xil seed) ishga
tushirganda manzara teskari bo'ladi:

| Funksiya (30D) | TEMOA_V10 | L-SHADE | jSO | LSHADE-cnEpSin | CMA-ES |
|---|---|---|---|---|---|
| Schwefel | ~2.2e+03 | **1.5e-01** | 6.8e-01 | 3.2e+00 | 4.2e+03 |
| NoisyRastrigin | ~3.2e+01 | **1.7e+00** | 5.6e+00 | 5.0e+00 | 6.6e+01 |
| RotatedElliptic | ~2.7e+04 | 9.5e+02 | 7.5e+01 | **3.5e-04** | 8.0e+06 |
| Rosenbrock | ~6.2e-01 | 1.2e+01 | 1.2e+01 | 1.4e+01 | **1.5e-27** |
| BentCigar | ~1.3e-22 | 1.0e-14 | 2.8e-18 | 1.3e-17 | **1.3e-29** |
| Ackley | ~8.7e-15 | 7.3e-12 | 8.4e-13 | 1.4e-13 | 1.5e-14 |

TEMOA Schwefel'da L-SHADE'dan **~15 000 marta**, NoisyRastrigin'da **~19 marta**,
RotatedElliptic'da **~8e7 marta** yomon.

> **Xulosa:** oldingi jadvaldagi "to'liq ustunlik" zaif bazaviy to'plam
> artefakti. Hozirgi holatda maqola Q1 jurnalida saqlanib qolmaydi.

---

## 4. Qaysi qism yutqazayapti: ablatsiya natijalari

Yutqazayotgan masalalarda komponentlarni birma-bir o'chirib ko'rildi
(11 run, mediana; batafsil: `docs/ABLATION.md`):

| Konfiguratsiya | Schwefel 30D | Schwefel 50D | Schwefel 100D | NoisyRast. 50D | NoisyRast. 100D |
|---|---|---|---|---|---|
| BASE (hozirgi) | 2428.0 | 5704.9 | 14865.8 | 58.7 | 144.4 |
| − CR_FLOOR | 2191.2 | 6336.7 | 12714.4 | 58.7 | 147.3 |
| − (1+1)-ES quyrug'i | 2428.0 | 5704.9 | 14865.8 | 58.7 | 144.4 |
| − eigen crossover | 1895.1 | 4323.6 | 11569.1 | 53.8 | 120.5 |
| operator portfeli = bir tekis | 1914.9 | 5606.4 | 12321.0 | 52.8 | 139.4 |
| **faqat current-to-pbest (portfel yo'q)** | **0.0** | **745.5** | **7422.5** | **39.1** | 146.5 |
| N_init = 18·D | 1717.4 | 5369.5 | 10028.3 | 56.8 | 121.7 |
| DE (raqib) | 2031.1 | 4282.0 | 10703.8 | 35.0 | 109.4 |

Ikkita aniq xulosa:

1. **Metafora asosidagi operator portfeli — asosiy yo'qotish manbai.**
   GWO lider-DE, WOA spirali va HHO Levy operatorlarini o'chirib, faqat
   `current-to-pbest-w/1` qoldirilganda Schwefel 30D xatosi **2428 → 0.0**
   ga tushadi. Sabab: probability matching krediti muvaffaqiyat *ulushi*ga
   asoslangan; ko'p ekstremumli relyefda eng "xavfsiz" operator eng yuqori
   ulushni oladi va portfel ekspluatatsiyaga qulaydi, tadqiqot operatorlari
   esa `P_MIN = 0.05` ga siqiladi.
2. **(1+1)-ES quyrug'i umuman ishlamaydi.** Beshta instansiyada ham natija
   BASE bilan raqamma-raqam bir xil. Byudjetning 5% i behuda sarflanadi.

Qo'shimcha: eigen-crossover separabel masalalarda zarar qiladi, lekin
RotatedElliptic'da hal qiluvchi darajada foydali — demak uni o'chirish emas,
**to'g'ri moslashtirish** kerak (hozirgi `[0.1, 0.9]` cheklovi uni butunlay
o'chira olmaydi).

---

## 5. Metodologik kamchiliklar (Q1 retsenzenti so'raydi)

1. **Benchmark standart emas.** O'z-o'zidan yasalgan 12 ta funksiya o'rniga
   CEC-2017 / CEC-2022 to'plami talab qilinadi.
2. **Byudjet kichik.** `3000 x D` o'rniga CEC standarti `10000 x D`.
3. **Run soni.** CEC protokoli 51 run, hozir 30.
4. **Post-hoc test yo'q edi.** Endi `friedman_posthoc.csv` sifatida qo'shildi.
5. **Ablatsiya tadqiqoti yo'q edi.** Gibrid maqolada har bir komponentning
   hissasi ko'rsatilishi shart.
6. **Hisoblash murakkabligi tahlili yo'q** (CEC uslubidagi T0/T1/T2 jadvali).
7. **Parametrlarga sezgirlik tahlili yo'q.**
