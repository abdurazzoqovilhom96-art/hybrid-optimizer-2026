# ACE-SHADE: bajarish promti

> Bu hujjat o'zini o'zi yetarli. Kontekst yo'qolsa shu fayldan davom
> ettiriladi. Holat: `results/STATUS_V2.md`.

## 0. Mandat va qat'iy qoidalar

**Mandat.** "Buni qilib bo'lmaydi" — natija emas. Har bir o'lchangan
zaiflik uchun matematik asoslangan yechim quriladi va sinaladi. To'xtash
sharti faqat o'lchov natijasi.

**Buzilmaydigan qoidalar:**

1. Sozlash **faqat BBOB** da, baholash **faqat CEC-2022** da. Bu
   to'plamlar hech qachon aralashmaydi.
2. Parametrlar B2 bosqichida muzlatiladi va keyin **o'zgartirilmaydi**.
3. Har bir yangi komponent ablatsiya bilan asoslanadi.
4. Barcha algoritmlar bir xil: seed sxemasi, chegara ishlovi, baholash
   hisoblagichi, to'xtash sharti.
5. Natijalar bezalmaydi. Mag'lubiyat raqami bilan yoziladi.
6. Faqat `claude/youthful-pascal-sjxqnn` branchiga push.
7. Har qadamdan keyin darhol commit + push (konteyner vaqtinchalik).

## 1. Qарорlar (tasdiqlangan)

| Element | Qiymat |
|---|---|
| Nom | **ACE-SHADE** (Adaptive Covariance-Ensemble SHADE) |
| Baholash to'plami | **CEC-2022** (12 funksiya, D = 10 va 20) |
| Sozlash to'plami | **BBOB** (COCO), baholash to'plamidan butunlay ajratilgan |
| Raqobatchilar | L-SHADE, jSO, LSHADE-cnEpSin, LSHADE-SPACMA, LSHADE-RSP, AGSK, NL-SHADE-RSP, DE, CMA-ES |
| Boshlanish | B0 — noldan kod yozish |

### CEC-2022 protokoli (aniq)

- 12 funksiya, qidiruv sohasi `[-100, 100]^D`
- `D = 10` → `MaxFES = 200 000`; `D = 20` → `MaxFES = 1 000 000`
- 30 ta mustaqil run
- Xato `< 1e-8` → 0 (CEC konventsiyasi)

**Funksiya kategoriyalari** (3-bo'limdagi tahlil uchun majburiy):

| Kategoriya | Funksiyalar | Nimani sinaydi |
|---|---|---|
| Bir ekstremumli | F1 | Yaqinlashish tezligi, shartlanganlik |
| Asosiy ko'p ekstremumli | F2–F5 | Global qidiruv |
| Gibrid | F6–F8 | Aralash tuzilma, qisman separabellik |
| Kompozitsiya | F9–F12 | Umumiy mustahkamlik |

## 2. ACE-SHADE: to'liq matematik spetsifikatsiya

### 2.1 Holat o'zgaruvchilari

```
P        populyatsiya, hajmi N_t        A        arxiv
M_F[b], M_CR[b]   b = 0 (jSO), 1 (L-SHADE), har biri H = 6 katak
p_bank, c_bank    bank ehtimolligi va krediti
p_eig,  c_eig     eigen-bazis ehtimolligi va krediti
p_cma,  c_cma     CMA tarmog'i ulushi va krediti
C, p_c            to'plamli kovariatsiya va evolyutsiya yo'li
B, d              C = B diag(d^2) B^T  (dangasa yangilanish)
nu                shovqin bahosi
```

### 2.2 Boshlang'ich qiymatlar

```
N_init = round(r_N * D),  N_min = 4,  H = 6,  A_r = 2.6
bank 0 (jSO):     M_F = 0.3,  M_CR = 0.8,  terminal katak (0.9, 0.9)
bank 1 (L-SHADE): M_F = 0.5,  M_CR = 0.5
p_bank = p_eig = 0.5,   p_cma = 0.1
C = I,  p_c = 0,  nu = 0
```

`r_N`, `kappa`, `lambda_nu`, `eta`, `g_nu` — BBOB da sozlanadi (B1).

### 2.3 To'plamli kovariatsiya (N1 — asosiy yangilik)

Joriy kod har avlodda `np.cov(pop[:N/2])` oladi: `n = 3D` namunadan `D`
o'lchamda, spektral xato `O(sqrt(D/n)) ~ 58%`. Buning o'rniga:

```
mu       = floor(N_t / 2)
w_k      = ln(mu + 0.5) - ln(k),   k = 1..mu,   normallashtirilgan
mu_eff   = 1 / sum(w_k^2)
c_c      = (4 + mu_eff/D) / (D + 4 + 2 mu_eff/D)
c_1      = 2 / ((D + 1.3)^2 + mu_eff)
c_mu     = min(1 - c_1, 2 (mu_eff - 2 + 1/mu_eff) / ((D + 2)^2 + mu_eff))
```

Har avlodda **qabul qilingan** avlodlarning eng yaxshi `mu` tasi bo'yicha:

```
m_old <- m
m     <- sum_k w_k * x_(k)
y_w   <- (m - m_old) / sigma
p_c   <- (1 - c_c) p_c + sqrt(c_c (2 - c_c) mu_eff) * y_w
y_k   <- (x_(k) - m_old) / sigma
C     <- (1 - c_1' - c_mu') C + c_1' p_c p_c^T + c_mu' sum_k w_k y_k y_k^T
```

bu yerda shovqin damplashi bilan `c_1' = c_1/(1+nu)`, `c_mu' = c_mu/(1+nu)`.

**Qadam hajmi CSA bilan emas, populyatsiya tarqalishidan olinadi:**

```
sigma = mean_j( std_i( P[i, j] ) )
```

*Sabab:* CSA Gauss namunasini faraz qiladi; bizda avlodlarning bir qismi
DE mutatsiyasidan keladi, shuning uchun CSA beqaror bo'lishi mumkin.
Populyatsiya tarqalishi taqsimotdan mustaqil, qo'shimcha parametrsiz va
CMA tarmog'ini DE populyatsiyasining miqyosiga bog'laydi.

**Dangasa xos yoyilma:** `fes - eigeneval > 1 / (10 D (c_1 + c_mu))`
bo'lganda qayta hisoblanadi. `C` simmetriklanadi
(`C = (C + C^T)/2`), xos qiymatlar `>= 1e-20 * max` gacha qirqiladi.
Narx `O(D^3)` amortizatsiya qilinib `O(D^2)` ga tushadi.

**Muhim metodologik izoh (maqolada ochiq yoziladi):** `y_k` faqat CMA
tarmog'idan emas, **barcha qabul qilingan** avlodlardan olinadi. Bu
darslikdagi CMA baholagichi emas — bu "muvaffaqiyatli qadamlarning
kovariatsiyasi". Bizga aynan shu kerak, chunki `C` crossover bazisi
sifatida ishlatiladi. B4 ablatsiyasida faqat CMA-tarmoq variantı bilan
taqqoslanadi.

### 2.4 Bitta model, ikki iste'molchi (N1 davomi)

**(a) Eigen-bazisda crossover** — `p_eig` ehtimollik bilan:

```
U_i = ( binom_cross( V_i @ B, x_i @ B, CR_i ) ) @ B^T
```

(satr vektorlar uchun `x @ B` = `B^T x` proyeksiyasi — joriy koddagi
shakl **to'g'ri**, o'zgartirilmaydi.)

**(b) CMA taqsimotidan namuna** — `p_cma_eff` ulush bilan:

```
z ~ N(0, I);   U_i = m + sigma * ( B @ (d * z) )
p_cma_eff = p_cma / (1 + lambda_nu * nu)
```

Alohida quyruq fazasi **yo'q**. `TAIL`, `TAIL_FRAC`, `TAIL_DIV` butunlay
olib tashlanadi.

### 2.5 DE yadrosi (saqlanadi)

```
p_num  = max(2, round((P_max - (P_max - P_min) t) N_t))
x_pbest ~ top p_num
r1: RSP vaznlari w_k = K (N - k)/N + 1,  r1 != i
r2 ~ P u A,   r2 != i, r2 != r1
V_i = x_i + Fw_i (x_pbest - x_i) + F_i (x_r1 - x_r2)
```

Chegara: midpoint-target (ikkala tarmoq uchun bir xil).

### 2.6 Parametr rejimi ansambli (saqlanadi — o'lchangan kuch)

```
bank_i ~ Bernoulli(1 - p_bank)       # 0 = jSO, 1 = L-SHADE
CR_i = clip(N(mu_CR, 0.1), 0, 1);  mu_CR < 0 -> CR_i = 0
F_i  ~ Cauchy(mu_F, 0.1), (0, 1] ga qirqiladi
bank 0 va t < 0.6:  F_i <- min(F_i, 0.7)
bank 0:  Fw_i = F_i * (0.7 | 0.8 | 1.2)   t < 0.2 | t < 0.4 | aks holda
bank 1:  Fw_i = F_i
```

`rho` prior "issiq start" **olib tashlanadi** — matematik asosi yo'q.

### 2.7 Shovqin bahosi va uning uch ta'siri (N3)

Har `g_nu` avlodda joriy eng yaxshi nuqta qayta baholanadi:

```
s   = IQR(fitness) + eps
nu <- (1 - alpha) nu + alpha * |f_1(x*) - f_2(x*)| / s
```

| Qayerda | Formula |
|---|---|
| Tanlov | qabul: `f(u_i) <= f(x_i) - kappa * nu * s` |
| Kovariatsiya | `c_1' = c_1/(1+nu)`, `c_mu' = c_mu/(1+nu)` |
| CMA tarmog'i | `p_cma_eff = p_cma / (1 + lambda_nu * nu)` |

`nu = 0` bo'lganda tanlov sharti `f(u) <= f(x)` ga aylanadi — ya'ni
shovqinsiz masalada joriy xulq aynan saqlanadi.

### 2.8 Birlashgan kredit arbitraji (N2)

Uchala moslashuvchan ulush (`p_bank`, `p_eig`, `p_cma`) **bir xil**
qoida bilan yangilanadi:

```
gain_i = max(f(x_i) - f(u_i), 0)  (faqat yaxshilanganlar)
har bir guruh g uchun:
    share_g  = |g| / N_t
    FIR_g    = (sum_{i in g} gain_i / sum gain) / share_g
    c_g     <- (1 - eta) c_g + eta * FIR_g
    p_g      = clip(c_g / sum c, 0.02, 0.98)
```

### 2.9 LPSR (saqlanadi)

```
N_{t+1} = max(N_min, round(N_init + (N_min - N_init) * fes / max_fes))
```

## 3. Chiqish: struktura saqlanadi + kategoriya tahlili

### 3.1 Avvalgi struktura o'zgarishsiz qoladi

Eksperiment tugagach **avtomatik** hosil bo'ladi (joriy `cba_shade.py`
dagi kabi, `out_dir` ostida):

```
tables/summary_mean_std.csv        tables/friedman_test.txt
tables/summary_results.csv         tables/friedman_posthoc.csv
tables/summary_table.tex           tables/wilcoxon_pvalues.csv
tables/overall_average_ranks.csv   tables/win_tie_loss.csv
tables/complexity.csv              figures/Conv_<F>_<D>.png
raw_data/full_raw_results.csv      raw_data/curves.npz
```

`PRECISION_FLOOR = 1e-8`, xom qiymatlar `raw_data/` da o'zgarishsiz.
Bo'lak/birlashtirish mexanizmi (`MERGE`, `OUT_DIR`, `DIMS`, `RUNS`)
va GitHub Actions ish oqimi ham saqlanadi.

### 3.2 YANGI: kategoriya bo'yicha tahlil (majburiy)

Quyidagilar ham **avtomatik** hosil bo'ladi:

```
tables/category_ranks.csv          har kategoriya x har algoritm: o'rtacha rank
tables/category_wtl_<rival>.csv    har raqib uchun kategoriya bo'yicha +/=/-
tables/category_summary.md         to'liq matnli tahlil
tables/dimension_category.csv      kategoriya x o'lcham kesimi
```

`category_summary.md` har bir kategoriya uchun quyidagi jadvalni
o'z ichiga oladi:

| Kategoriya | Raqib | + | = | − | Bizning o'rtacha rank | Uning ranki | Holat |
|---|---|---|---|---|---|---|---|
| Bir ekstremumli (F1) | L-SHADE | | | | | | YUTDIK / TENG / YUTQAZDIK |
| ... | ... | | | | | | |

**Holat qoidasi:** `YUTDIK` faqat Holm tuzatilgan `p < 0.05` bo'lganda.
O'rtacha kichikroq bo'lishi yetarli emas.

Har bir kategoriya uchun matnda quyidagilar yoziladi:
- Qaysi funksiyada aniq yutdik va qanchaga (raqam bilan)
- Qaysi funksiyada yutqazdik va qanchaga (raqam bilan)
- Sabab bo'yicha taxmin (qaysi komponent ishladi/ishlamadi)

Shovqin komponenti (N3) uchun: CEC-2022 da shovqinli funksiya **yo'q**,
shuning uchun N3 ni tasdiqlash uchun **BBOB shovqinli to'plami**
(`f101`–`f130`) da alohida qo'shimcha tajriba o'tkaziladi va
`tables/noise_study.csv` sifatida saqlanadi. Bu maqolada alohida
bo'lim bo'ladi.

## 4. Bosqichlar

| Bosqich | Mazmun | Chiqish |
|---|---|---|
| **B0** | `ace_shade.py`: ACE-SHADE + 9 raqobatchi + CEC-2022 + BBOB yuklagich + barcha jadval generatorlari + birlik testlari | kod, testlar yashil |
| **B1** | BBOB da `r_N`, `kappa`, `lambda_nu`, `eta`, `g_nu` sozlash (10D/20D, 15 run) | `results/tuning_v2/` |
| **B2** | Parametrlarni muzlatish va e'lon qilish | `docs/ABLATION_V2.md` |
| **B3** | CEC-2022 to'liq: 12 F x 2 D x 30 run x 10 alg | `results/v2/` |
| **B4** | Ablatsiya: N1, N2, N3 alohida o'chirilgan | `results/ablation_v2/` |
| **B5** | BBOB shovqinli to'plamida N3 tasdiqlash | `results/noise_v2/` |
| **B6** | `docs/RESULTS_V2.md` + kategoriya tahlili | hisobot |

### B0 birlik testlari (majburiy, o'tmasa keyingi bosqichga o'tilmaydi)

1. `C` simmetrik va musbat yarim aniq qoladi (1000 avlod)
2. Xos yoyilma dangasa davri to'g'ri ishlaydi; `B @ diag(d^2) @ B.T ~ C`
3. `nu = 0` da tanlov sharti aynan `f(u) <= f(x)` ga teng
4. Baholash hisoblagichi `MaxFES` dan oshmaydi (barcha algoritmlar)
5. Chegara ishlovi barcha nuqtalarni `[lb, ub]` ichida saqlaydi
6. Seed bir xil bo'lganda natija takrorlanadi
7. CEC-2022 funksiyalari ma'lum optimumda `0` beradi

## 5. Ma'lumot fayllari bo'yicha ehtiyot chorasi

CEC-2022 rasmiy siljitish/aylantirish fayllari kerak. Tartib:

1. Rasmiy paketni olishga urinish (COCO/CEC omborlari).
2. **Agar olinmasa:** siljitish vektorlari va aylantirish matritsalari
   qat'iy belgilangan seed bilan generatsiya qilinadi va
   `data/cec2022/` da saqlanadi. Maqolada **ochiq yoziladi**: bu
   protokolga sodiq qayta amalga oshirish, rasmiy fayllar bilan
   bit-identik emas. Barcha algoritmlar bir xil instansiyalarni
   ko'rgani uchun taqqoslash haqiqiy bo'lib qoladi, lekin adabiyotdagi
   raqamlar bilan bevosita solishtirib bo'lmaydi.

BBOB uchun `cocoex` paketini o'rnatishga urinish; bo'lmasa BBOB
funksiyalari ta'rifidan amalga oshiriladi.

## 6. Xavflar (oldindan qayd etilgan)

| Xavf | Javob |
|---|---|
| `y_k` ni DE avlodlaridan olish darslik CMA baholagichi emas | B4 da faqat CMA-tarmoq varianti bilan taqqoslanadi |
| `10000*D` emas, CEC-2022 byudjeti kattaroq — raqiblar kuchayadi | Halol taqqoslash narxi; oldindan qabul qilingan |
| 20D da `MaxFES = 10^6` — hisoblash qimmat | Funksiya bo'yicha shardlash (12 ish) |
| Rasmiy ma'lumot fayllari topilmasligi | 5-bo'lim tartibi |
| Yangi komponentlar bir-birini bekor qilishi | B4 har birini alohida o'lchaydi |
