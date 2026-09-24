# ACE-SHADE: yakuniy bajarish promti

> **Bu oxirgi versiya.** Ko'lam yopildi. Quyidagilar o'zgarmaydi:
> maydon, komponentlar, raqobatchilar ro'yxati, muvaffaqiyat mezoni.
> Yangi komponent, yangi benchmark yoki yangi ilova qo'shilmaydi.
>
> Holat: `results/STATUS_V2.md`.

## 0. Maqsad va to'xtash sharti

**Maqsad.** Oldingi ishlardan o'lchanadigan darajada yuqori o'rin
egallaydigan gibrid algoritm va uni hujjatlashtirgan Q1 darajasidagi
hisobot.

**Muvaffaqiyat mezoni (oldindan belgilangan, keyin o'zgartirilmaydi):**

| Shart | Talab |
|---|---|
| S1 | ACE-SHADE CEC-2022 da eng yaxshi o'rtacha rank |
| S2 | Holm tuzatilgan post-hoc: L-SHADE, jSO, LSHADE-cnEpSin ga nisbatan `p < 0.05` |
| S3 | Ablatsiya: N1 va N2 ning har biri alohida hissa qo'shadi |

**To'xtash sharti.** B3 tugagach natija qanday bo'lsa shunday yoziladi.
Qayta sozlash yo'q, maydon almashtirish yo'q, komponent qo'shish yo'q.
Bir marta o'lchaymiz, bir marta yozamiz.

S2 NL-SHADE-RSP ga nisbatan bajarilmasa — bu **muvaffaqiyatsizlik emas**.
"CEC-2021 g'olibi bilan raqobatbardosh, 2014–2018 oilasidan statistik
jihatdan ustun" — Q1 uchun yetarli natija va shunday yoziladi.

## 1. Qat'iy qoidalar

1. Sozlash faqat **CEC-2017**, baholash faqat **CEC-2022**. Kesishmaydi.
2. Parametrlar B2 da muzlatiladi, keyin o'zgartirilmaydi.
3. Har bir komponent ablatsiya bilan asoslanadi; asoslanmagani olib
   tashlanadi.
4. Barcha algoritmlar bir xil: seed, chegara ishlovi, baholash
   hisoblagichi, to'xtash sharti.
5. Natijalar bezalmaydi.
6. Faqat `claude/youthful-pascal-sjxqnn` branchiga push; har qadamdan
   keyin darhol commit.

## 2. Ko'lam: nima ichida, nima tashqarida

| Ichida | Tashqarida va nega |
|---|---|
| CEC-2022 (12 F, D = 10/20, 30 run) | `bbob-noisy` — raqiblarda shovqin mexanizmi yo'q, g'alaba hech narsa isbotlamaydi |
| N1 — to'plamli kovariatsiya | N3 — shovqin damping: CEC-2022 da shovqinli funksiya yo'q, asoslab bo'lmaydi → **olib tashlanadi** |
| N2 — parametr rejimi ansambli | RL ilovasi — alohida ish, bu maqolaga kirmaydi |
| 7 raqobatchi | AGSK, UH-CMA-ES, DE — qayta yozish xavfi yoki hissa qo'shmaydi |

**N3 olib tashlanishi qoida 3 ning bevosita natijasi.** Shovqin
komponentini tasdiqlaydigan funksiya yo'q ekan, uni kiritish o'z
metodologiyamizni buzish bo'lardi. Bu topilma `docs/RESULTS.md` da
qolgan zaiflik sifatida qayd etiladi va kelajakdagi ish bo'ladi.

## 3. Raqobatchilar (7 ta, o'zgarmaydi)

| Algoritm | Roli |
|---|---|
| NL-SHADE-RSP | CEC-2021 g'olibi — eng kuchli raqib |
| LSHADE-SPACMA | eng yaqin gibrid (DE + CMA) |
| LSHADE-cnEpSin | eigen-crossover manbai — N1 shunga qarshi |
| LSHADE-RSP | RSP manbai |
| jSO | 0-bank manbai — N2 shunga qarshi |
| L-SHADE | 1-bank manbai — N2 shunga qarshi |
| CMA-ES | kovariatsiya nazorati — N1 shunga qarshi |

Har biri maqoladagi psevdokod bo'yicha yoziladi va CEC-2017 da
e'lon qilingan tartib bilan sanity-test qilinadi. Nomuvofiqlik topilsa
hisobotda ochiq yoziladi.

## 4. ACE-SHADE: spetsifikatsiya

### 4.1 Holat

```
P (hajmi N_t), A (arxiv)
M_F[b], M_CR[b]   b = 0 (jSO), 1 (L-SHADE), H = 6 katak
p_bank, p_eig, p_cma va kreditlari c_*
C (to'plamli kovariatsiya), p_c (evolyutsiya yo'li)
B, d:  C = B diag(d^2) B^T
```

### 4.2 Boshlang'ich

```
N_init = round(r_N * D), N_min = 4, H = 6, A_r = 2.6
bank 0 (jSO):     M_F = 0.3, M_CR = 0.8, terminal katak (0.9, 0.9)
bank 1 (L-SHADE): M_F = 0.5, M_CR = 0.5
p_bank = p_eig = 0.5, p_cma = 0.1, C = I, p_c = 0
```

Sozlanadigan (faqat ikkita): `r_N`, `eta`.

### 4.3 N1 — to'plamli kovariatsiya

Joriy kod har avlodda `np.cov(pop[:N/2])` oladi: `n = 3D` namunadan `D`
o'lchamda, spektral xato `O(sqrt(D/n)) ~ 58%`. O'rniga:

```
mu     = floor(N_t / 2)
w_k    = ln(mu + 0.5) - ln(k),  k = 1..mu,  normallashtirilgan
mu_eff = 1 / sum(w_k^2)
c_c    = (4 + mu_eff/D) / (D + 4 + 2 mu_eff/D)
c_1    = 2 / ((D + 1.3)^2 + mu_eff)
c_mu   = min(1 - c_1, 2 (mu_eff - 2 + 1/mu_eff) / ((D + 2)^2 + mu_eff))

m_old <- m;  m <- sum_k w_k x_(k)        # qabul qilingan eng yaxshi mu ta
y_w    = (m - m_old) / sigma
p_c   <- (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) y_w
y_k    = (x_(k) - m_old) / sigma
C     <- (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu sum_k w_k y_k y_k^T
sigma  = mean_j( std_i( P[i,j] ) )       # CSA emas: aralash manbada beqaror
```

Dangasa xos yoyilma: `fes - eigeneval > 1/(10 D (c_1 + c_mu))` da;
`C <- (C + C^T)/2`; xos qiymatlar `>= 1e-20 * max` gacha qirqiladi.

**Ochiq izoh:** `y_k` barcha qabul qilingan avlodlardan olinadi. Bu
darslikdagi CMA baholagichi emas — bu *muvaffaqiyatli qadamlarning
kovariatsiyasi*, crossover bazisi uchun aynan shu kerak. B4 da
faqat-CMA-tarmoq varianti bilan taqqoslanadi.

**Ikki iste'molchi:**
- eigen-crossover (`p_eig`): `U_i = binom_cross(V_i @ B, x_i @ B, CR_i) @ B^T`
- CMA namunasi (`p_cma`): `U_i = m + sigma (B @ (d * z))`, `z ~ N(0,I)`

Alohida quyruq fazasi yo'q: `TAIL`, `TAIL_FRAC`, `TAIL_DIV` olib tashlanadi.

### 4.4 N2 — parametr rejimi ansambli (o'lchangan kuch)

```
bank_i ~ Bernoulli(1 - p_bank)
CR_i = clip(N(mu_CR, 0.1), 0, 1);  mu_CR < 0 -> 0
F_i  ~ Cauchy(mu_F, 0.1) -> (0, 1]
bank 0, t < 0.6:  F_i <- min(F_i, 0.7)
bank 0:  Fw_i = F_i * (0.7 | 0.8 | 1.2)   t < 0.2 | t < 0.4 | aks holda
bank 1:  Fw_i = F_i
```

`rho` prior "issiq start" olib tashlanadi — matematik asosi yo'q.

### 4.5 DE yadrosi va LPSR (saqlanadi)

```
p_num = max(2, round((P_max - (P_max - P_min) t) N_t))
r1: RSP vaznlari w_k = K (N-k)/N + 1,  r1 != i
r2 ~ P u A,  r2 != i, r1
V_i = x_i + Fw_i (x_pbest - x_i) + F_i (x_r1 - x_r2)
N_{t+1} = max(N_min, round(N_init + (N_min - N_init) fes / max_fes))
```
Chegara: midpoint-target, ikkala tarmoq uchun bir xil.

### 4.6 Birlashgan kredit (N2 mexanizmi, uchala ulush uchun)

```
gain_i = max(f(x_i) - f(u_i), 0)
share_g = |g| / N_t
FIR_g   = (sum_{i in g} gain_i / sum gain) / share_g
c_g    <- (1 - eta) c_g + eta * FIR_g
p_g     = clip(c_g / sum c, 0.02, 0.98)
```

## 5. Chiqish (avvalgi struktura + kategoriya tahlili)

Eksperiment tugagach **avtomatik**:

```
tables/summary_mean_std.csv       tables/friedman_test.txt
tables/summary_results.csv        tables/friedman_posthoc.csv
tables/summary_table.tex          tables/wilcoxon_pvalues.csv
tables/overall_average_ranks.csv  tables/win_tie_loss.csv
tables/complexity.csv             figures/Conv_<F>_<D>.png
raw_data/full_raw_results.csv     raw_data/curves.npz
```

`PRECISION_FLOOR = 1e-8`; bo'lak/birlashtirish mexanizmi va Actions
ish oqimi saqlanadi.

**Kategoriya tahlili (majburiy):**

```
tables/category_ranks.csv     tables/category_wtl_<rival>.csv
tables/category_summary.md    tables/dimension_category.csv
```

Kategoriyalar: bir ekstremumli (F1) / asosiy ko'p ekstremumli (F2–F5) /
gibrid (F6–F8) / kompozitsiya (F9–F12), har biri D = 10 va 20 kesimida.

`category_summary.md` da har kategoriya x har raqib uchun:

| Kategoriya | Raqib | + | = | − | Bizning rank | Uning ranki | Holat |

**Holat qoidasi:** `YUTDIK` faqat Holm tuzatilgan `p < 0.05` bo'lganda.
Har kategoriya uchun matnda: qaysi funksiyada yutdik/yutqazdik, qancha
farq bilan (raqam), va qaysi komponent sabab bo'lgani haqida taxmin.

## 6. Bosqichlar va narx (o'lchangan o'tkazuvchanlik: 17 500 FES/yadro-sek)

| Bosqich | Mazmun | Actions daq |
|---|---|---|
| **B0** | `ace_shade.py`: ACE-SHADE + 7 raqobatchi + CEC-2017/2022 yuklagichlari + jadval generatorlari + 8 birlik testi | — |
| **B1** | CEC-2017 da `r_N`, `eta` sozlash (10 F, D=10, 15 run) | 43 |
| **B2** | Parametrlarni muzlatish va e'lon qilish | — |
| **B3** | CEC-2022: 12 F x 2 D x 30 run x 8 alg (12 shard, eng uzuni 69 daq) | 823 |
| **B4** | Ablatsiya: to'liq / N1 o'chiq / N2 o'chiq / faqat-CMA-tarmoq | 411 |
| **B5** | `docs/RESULTS_V2.md` + kategoriya tahlili | — |
| | **JAMI** | **~1 280** |

Devor vaqti: eng uzun shard 69 daqiqa; parallel bajarilganda butun
hisoblash **~2 soat** ichida tugaydi.

### B0 birlik testlari (o'tmasa keyingi bosqichga o'tilmaydi)

1. `C` simmetrik va musbat yarim aniq (1000 avlod)
2. `B diag(d^2) B^T ~ C`; dangasa davr to'g'ri
3. Baholash hisoblagichi `MaxFES` dan oshmaydi (barcha algoritmlar)
4. Chegara ishlovi barcha nuqtalarni `[lb, ub]` ichida saqlaydi
5. Bir xil seed → bir xil natija
6. CEC-2017 va CEC-2022 optimumda rasmiy bias beradi
7. Har bir raqobatchi CEC-2017 da kutilgan tartibni beradi
8. LPSR `N_min` dan past tushmaydi

## 7. Xavflar

| Xavf | Javob |
|---|---|
| NL-SHADE-RSP ni to'g'ri yozish | Psevdokod bo'yicha; sanity-test; nomuvofiqlik ochiq yoziladi |
| S2 NL-SHADE-RSP ga nisbatan bajarilmasligi | Oldindan qabul qilingan — 0-bo'limga qarang |
| `y_k` ni DE avlodlaridan olish | B4 ablatsiyasi o'lchaydi |
| D = 20 da `MaxFES = 10^6` | 12 shard, eng uzuni 69 daqiqa |
